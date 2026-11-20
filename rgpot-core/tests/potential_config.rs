use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use rgpot_core::Potentials_capnp::potential_config;

#[test]
fn morse_arm_round_trips() {
    let mut message = Builder::new_default();
    {
        let config = message.init_root::<potential_config::Builder>();
        let mut morse = config.init_morse();
        morse.set_de(1.25);
        morse.set_a(1.5);
        morse.set_re(2.75);
        morse.set_cutoff(8.0);
    }
    let encoded = serialize::write_message_to_words(&message);

    let mut bytes = encoded.as_slice();
    let reader = serialize::read_message_from_flat_slice(&mut bytes, ReaderOptions::new())
        .expect("read PotentialConfig");
    let config = reader
        .get_root::<potential_config::Reader>()
        .expect("PotentialConfig root");
    let morse = match config.which().expect("known PotentialConfig arm") {
        potential_config::Which::Morse(morse) => morse.expect("morse arm"),
        _ => panic!("PotentialConfig decoded to another arm"),
    };
    assert_eq!(morse.get_de(), 1.25);
    assert_eq!(morse.get_a(), 1.5);
    assert_eq!(morse.get_re(), 2.75);
    assert_eq!(morse.get_cutoff(), 8.0);
}

#[test]
fn morse_arm_defaults_are_the_platinum_parameters() {
    let mut message = Builder::new_default();
    message
        .init_root::<potential_config::Builder>()
        .init_morse();
    let config = message
        .get_root_as_reader::<potential_config::Reader>()
        .expect("PotentialConfig root");
    let potential_config::Which::Morse(morse) = config.which().expect("known arm") else {
        panic!("PotentialConfig decoded to another arm");
    };
    let morse = morse.expect("morse arm");
    assert_eq!(morse.get_de(), 0.7102);
    assert_eq!(morse.get_a(), 1.6047);
    assert_eq!(morse.get_re(), 2.897);
    assert_eq!(morse.get_cutoff(), 9.5);
}

#[test]
fn muller_brown_lists_are_empty_unless_set() {
    let mut message = Builder::new_default();
    message
        .init_root::<potential_config::Builder>()
        .init_muller_brown();
    let config = message
        .get_root_as_reader::<potential_config::Reader>()
        .expect("PotentialConfig root");
    let potential_config::Which::MullerBrown(params) = config.which().expect("known arm") else {
        panic!("PotentialConfig decoded to another arm");
    };
    let params = params.expect("mullerBrown arm");
    assert_eq!(params.get_amplitude().expect("amplitude").len(), 0);
    assert_eq!(params.get_a().expect("a").len(), 0);
    assert_eq!(params.get_b().expect("b").len(), 0);
    assert_eq!(params.get_c().expect("c").len(), 0);
    assert_eq!(params.get_x0().expect("x0").len(), 0);
    assert_eq!(params.get_y0().expect("y0").len(), 0);
}

#[test]
fn muller_brown_lists_round_trip() {
    let amplitude = [-400.0_f64, -200.0, -340.0, 30.0];
    let a = [-1.0_f64, -2.0, -6.5, 0.7];
    let b = [0.0_f64, 0.0, 11.0, 0.6];
    let c = [-10.0_f64, -10.0, -6.5, 0.7];
    let x0 = [1.0_f64, 0.25, -0.5, -1.0];
    let y0 = [0.0_f64, 0.5, 1.5, 1.25];
    let mut message = Builder::new_default();
    {
        let config = message.init_root::<potential_config::Builder>();
        let mut params = config.init_muller_brown();
        let mut out = params.reborrow().init_amplitude(4);
        for (index, value) in amplitude.iter().enumerate() {
            out.set(index as u32, *value);
        }
        let mut out = params.reborrow().init_a(4);
        for (index, value) in a.iter().enumerate() {
            out.set(index as u32, *value);
        }
        let mut out = params.reborrow().init_b(4);
        for (index, value) in b.iter().enumerate() {
            out.set(index as u32, *value);
        }
        let mut out = params.reborrow().init_c(4);
        for (index, value) in c.iter().enumerate() {
            out.set(index as u32, *value);
        }
        let mut out = params.reborrow().init_x0(4);
        for (index, value) in x0.iter().enumerate() {
            out.set(index as u32, *value);
        }
        let mut out = params.reborrow().init_y0(4);
        for (index, value) in y0.iter().enumerate() {
            out.set(index as u32, *value);
        }
    }
    let encoded = serialize::write_message_to_words(&message);
    let mut bytes = encoded.as_slice();
    let reader = serialize::read_message_from_flat_slice(&mut bytes, ReaderOptions::new())
        .expect("read PotentialConfig");
    let config = reader
        .get_root::<potential_config::Reader>()
        .expect("PotentialConfig root");
    let potential_config::Which::MullerBrown(params) = config.which().expect("known arm") else {
        panic!("PotentialConfig decoded to another arm");
    };
    let params = params.expect("mullerBrown arm");
    let check = |list: capnp::Result<capnp::primitive_list::Reader<'_, f64>>, want: &[f64]| {
        let list = list.expect("list");
        assert_eq!(list.len() as usize, want.len());
        for (index, value) in want.iter().enumerate() {
            assert_eq!(list.get(index as u32), *value);
        }
    };
    check(params.get_amplitude(), &amplitude);
    check(params.get_a(), &a);
    check(params.get_b(), &b);
    check(params.get_c(), &c);
    check(params.get_x0(), &x0);
    check(params.get_y0(), &y0);
}

/// potentials-schema's arm ordinals, shared with gpridge and every other
/// consumer. A renumbered arm would decode as another backend on the wire.
const CONFIG_ARMS: [(&str, u16); 17] = [
    ("none", 0),
    ("nwchem", 1),
    ("cpmd", 2),
    ("metatomic", 4),
    ("lammps", 5),
    ("xtb", 6),
    ("tblite", 7),
    ("uma", 8),
    ("lj", 9),
    ("morse", 10),
    ("zbl", 11),
    ("d3", 12),
    ("d4", 13),
    ("mopac", 14),
    ("skala", 15),
    ("expr", 16),
    ("cp2k", 17),
];

/// The arms rgpot adds in CppCore/rgpot/rpc/local, after the upstream ones.
const LOCAL_ARMS: [(&str, u16); 4] = [
    ("composite", 18),
    ("d2", 19),
    ("d2cpmd", 20),
    ("mullerBrown", 21),
];

#[test]
fn union_arms_keep_the_upstream_ordinals() {
    use capnp::introspect::{Introspect, TypeVariant};
    use capnp::schema_capnp::field;

    let TypeVariant::Struct(raw) = potential_config::Owned::introspect().which() else {
        panic!("PotentialConfig is not a struct");
    };
    let schema = capnp::schema::StructSchema::new(raw);
    let arms = schema.get_union_fields().expect("PotentialConfig union");
    let mut seen = Vec::new();
    for arm in arms {
        let proto = arm.get_proto();
        let name = proto.get_name().expect("name").to_str().expect("utf-8");
        let ordinal = match proto.get_ordinal().which().expect("ordinal") {
            field::ordinal::Explicit(ordinal) => ordinal,
            field::ordinal::Implicit(()) => panic!("{name} ordinal must be explicit"),
        };
        seen.push((name.to_owned(), ordinal));
    }
    let want: Vec<(String, u16)> = CONFIG_ARMS
        .iter()
        .chain(LOCAL_ARMS.iter())
        .map(|(name, ordinal)| ((*name).to_owned(), *ordinal))
        .collect();
    assert_eq!(seen, want);
}
