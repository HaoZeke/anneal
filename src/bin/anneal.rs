//! `anneal` command line.
//!
//! ```text
//! anneal landfold [OPTIONS] BOOK...
//! ```
//!
//! `landfold` reads one or more minima books, folds them onto the plane
//! with [`anneal_core::landfold::landfold`], and writes one row per
//! minimum. Run `anneal landfold --help` for the options.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anneal_core::landfold::{
    self, BookDescriptors, LandfoldConfig, LandfoldMetric, Minimum, book_descriptors,
};

const USAGE: &str = "\
usage: anneal landfold [OPTIONS] BOOK...

Fold a minima book onto the plane and write one row per minimum.

A BOOK is a .min dump (one minimum per line: energy, then 3N coordinates)
or a multi-frame .xyz file. Several books are concatenated in order.
Each minimum becomes a DECAF class histogram on one shared codebook, and
minima are compared by the L1 distance of their histograms.

Options:
  -o, --out FILE          write here instead of standard output
      --format csv|json   output format (default: from --out, else csv)
      --descriptors       read each BOOK as CSV descriptor rows instead
      --dump-descriptors FILE
                          also write the descriptor rows as CSV
      --seed N            seed of the first landmark and eigensolve (0)
      --landmarks M       farthest-point landmarks (400)
      --sigma S           high-dimensional switch scale (auto)
      --sigma-scale F     auto sigma is F times the median landmark distance (0.5)
      --sigma-low S       low-dimensional switch scale (sigma)
      --high A,B          high-dimensional switch exponents (2,4)
      --low a,b           low-dimensional switch exponents (2,4)
      --grid G            pointwise grid points per axis, 0 to skip (21)
      --iterations K      L-BFGS iterations per stage (200)
      --mix M1,M2,...     metric-stress weight of each stage (1,0.5,0.25,0.1,0)
      --weight-power G    landmark weight is its Voronoi population to G (0.5)
      --metric l1|l2      row distance (l1)
      --threads T         worker threads (all cores)
  -q, --quiet             no timing summary on standard error
  -h, --help              this text

CSV columns: index,energy,x,y,landmark,family,community.
";

enum Format {
    Csv,
    Json,
}

struct Args {
    books: Vec<PathBuf>,
    out: Option<PathBuf>,
    format: Option<Format>,
    descriptors: bool,
    dump_descriptors: Option<PathBuf>,
    threads: Option<usize>,
    quiet: bool,
    config: LandfoldConfig,
}

fn pair(value: &str, flag: &str) -> Result<[f64; 2], String> {
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("{flag} takes two comma-separated numbers"));
    }
    Ok([number(parts[0], flag)?, number(parts[1], flag)?])
}

fn number<T: std::str::FromStr>(value: &str, flag: &str) -> Result<T, String> {
    value
        .trim()
        .parse()
        .map_err(|_| format!("{flag}: cannot read {value:?}"))
}

fn parse_args(raw: Vec<String>) -> Result<Option<Args>, String> {
    let mut args = Args {
        books: Vec::new(),
        out: None,
        format: None,
        descriptors: false,
        dump_descriptors: None,
        threads: None,
        quiet: false,
        config: LandfoldConfig::default(),
    };
    let mut it = raw.into_iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| it.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "-o" | "--out" => args.out = Some(PathBuf::from(value(&arg)?)),
            "--format" => {
                args.format = Some(match value(&arg)?.as_str() {
                    "csv" => Format::Csv,
                    "json" => Format::Json,
                    other => return Err(format!("--format: unknown format {other:?}")),
                })
            }
            "--descriptors" => args.descriptors = true,
            "--dump-descriptors" => args.dump_descriptors = Some(PathBuf::from(value(&arg)?)),
            "--seed" => args.config.seed = number(&value(&arg)?, &arg)?,
            "--landmarks" => args.config.landmarks = number(&value(&arg)?, &arg)?,
            "--sigma" => args.config.sigma = Some(number(&value(&arg)?, &arg)?),
            "--sigma-scale" => args.config.sigma_scale = number(&value(&arg)?, &arg)?,
            "--sigma-low" => args.config.sigma_low = Some(number(&value(&arg)?, &arg)?),
            "--high" => args.config.high = pair(&value(&arg)?, &arg)?,
            "--low" => args.config.low = pair(&value(&arg)?, &arg)?,
            "--grid" => args.config.grid = number(&value(&arg)?, &arg)?,
            "--iterations" => args.config.iterations = number(&value(&arg)?, &arg)?,
            "--mix" => {
                args.config.mix = value(&arg)?
                    .split(',')
                    .map(|v| number(v, "--mix"))
                    .collect::<Result<_, _>>()?
            }
            "--weight-power" => args.config.weight_power = number(&value(&arg)?, &arg)?,
            "--metric" => {
                args.config.metric = match value(&arg)?.as_str() {
                    "l1" => LandfoldMetric::L1,
                    "l2" => LandfoldMetric::L2,
                    other => return Err(format!("--metric: unknown metric {other:?}")),
                }
            }
            "--threads" => args.threads = Some(number(&value(&arg)?, &arg)?),
            "-q" | "--quiet" => args.quiet = true,
            flag if flag.starts_with('-') && flag != "-" => {
                return Err(format!("unknown option {flag}"));
            }
            _ => args.books.push(PathBuf::from(arg)),
        }
    }
    if args.books.is_empty() {
        return Err("landfold needs at least one BOOK".into());
    }
    Ok(Some(args))
}

fn read_descriptor_csv(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let text = std::fs::read_to_string(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let mut rows = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row: Result<Vec<f64>, _> = line
            .split([',', ' ', '\t'])
            .filter(|t| !t.is_empty())
            .map(str::parse::<f64>)
            .collect();
        match row {
            Ok(row) => rows.push(row),
            Err(_) if rows.is_empty() => continue,
            Err(err) => {
                return Err(format!("{}:{}: {err}", path.display(), line_no + 1));
            }
        }
    }
    Ok(rows)
}

fn write_descriptor_csv(path: &Path, rows: &[Vec<f64>]) -> Result<(), String> {
    let mut out = String::new();
    for row in rows {
        let line: Vec<String> = row.iter().map(|v| format!("{v:?}")).collect();
        out.push_str(&line.join(","));
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|err| format!("{}: {err}", path.display()))
}

fn run_landfold(args: Args) -> Result<(), String> {
    let started = Instant::now();
    let mut energy: Option<Vec<f64>> = None;
    let mut labels: Option<BookDescriptors> = None;
    let rows = if args.descriptors {
        let mut rows = Vec::new();
        for path in &args.books {
            rows.extend(read_descriptor_csv(path)?);
        }
        rows
    } else {
        let mut book: Vec<Minimum> = Vec::new();
        for path in &args.books {
            book.extend(landfold::read_book(path).map_err(|err| err.to_string())?);
        }
        energy = Some(book.iter().map(|m| m.energy).collect());
        let coordinates: Vec<Vec<f64>> = book.into_iter().map(|m| m.coordinates).collect();
        let descriptors = book_descriptors(&coordinates).map_err(|err| err.to_string())?;
        let rows = descriptors.rows.clone();
        labels = Some(descriptors);
        rows
    };
    let read = started.elapsed();
    if let Some(path) = &args.dump_descriptors {
        write_descriptor_csv(path, &rows)?;
    }
    let fold = landfold::landfold(&rows, &args.config).map_err(|err| err.to_string())?;
    let folded = started.elapsed() - read;
    let format = args.format.unwrap_or_else(|| match &args.out {
        Some(path) if path.extension().is_some_and(|e| e == "json") => Format::Json,
        _ => Format::Csv,
    });
    let family = labels.as_ref().map(|d| d.family.as_slice());
    let community = labels.as_ref().map(|d| d.community.as_slice());
    let text = match format {
        Format::Csv => landfold::to_csv(&fold, energy.as_deref(), family, community),
        Format::Json => {
            let mut json = landfold::to_json(&fold, energy.as_deref(), family, community);
            json.push('\n');
            json
        }
    };
    match &args.out {
        Some(path) => {
            std::fs::write(path, text).map_err(|err| format!("{}: {err}", path.display()))?
        }
        None => std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(|err| err.to_string())?,
    }
    if !args.quiet {
        eprintln!(
            "landfold: {} rows, {} landmarks, sigma {:.6}, stress {:.6e}, lambda1 {:.6e}, lambda2 {:.6e}, read {:.3} s, fold {:.3} s",
            fold.xy.len(),
            fold.landmarks.len(),
            fold.map.high().sigma(),
            fold.stress,
            fold.spectrum[0],
            fold.spectrum[1],
            read.as_secs_f64(),
            folded.as_secs_f64()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || matches!(raw[0].as_str(), "-h" | "--help" | "help") {
        print!(
            "usage: anneal <command> [OPTIONS]\n\ncommands:\n  landfold   fold a minima book onto the plane\n"
        );
        return ExitCode::SUCCESS;
    }
    let command = raw.remove(0);
    if command != "landfold" {
        eprintln!("anneal: unknown command {command:?}; try `anneal landfold --help`");
        return ExitCode::from(2);
    }
    let args = match parse_args(raw) {
        Ok(Some(args)) => args,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(err) => {
            eprintln!("anneal landfold: {err}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let threads = args.threads;
    let result = match threads {
        Some(t) => match rayon::ThreadPoolBuilder::new().num_threads(t).build() {
            Ok(pool) => pool.install(|| run_landfold(args)),
            Err(err) => Err(err.to_string()),
        },
        None => run_landfold(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("anneal landfold: {err}");
            ExitCode::FAILURE
        }
    }
}
