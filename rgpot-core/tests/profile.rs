use capnp::message::Builder;
use capnp::serialize;
use rgpot_core::profile::{
    decode_potential_result, encode_force_input, library_candidates, ProfileRequest,
};
use rgpot_core::Potentials_capnp::potential_result;

#[test]
fn candidates_are_prefix_parameterized() {
    let nwchem = library_candidates("nwchemc", Some("/vendor/lib/libnwchemc.so"));
    let cpmd = library_candidates("cpmdc", None);

    assert_eq!(nwchem[0], "/vendor/lib/libnwchemc.so");
    assert!(nwchem.iter().any(|path| path == "libnwchemc.so"));
    assert!(cpmd.iter().any(|path| path == "libcpmdc.so"));
    assert!(!cpmd.iter().any(|path| path.contains("nwchem")));
}

#[test]
fn force_input_codec_preserves_geometry_and_units() {
    let positions = [0.0, 0.1, 0.2, 0.9, 1.0, 1.1];
    let atomic_numbers = [8, 1];
    let box_matrix = [12.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 14.0];
    let encoded = encode_force_input(&ProfileRequest {
        positions: &positions,
        atomic_numbers: &atomic_numbers,
        box_matrix: Some(&box_matrix),
        length_unit: "angstrom",
        energy_unit: "eV",
    })
    .expect("encode ForceInput");

    let mut bytes = encoded.as_slice();
    let message =
        serialize::read_message_from_flat_slice(&mut bytes, capnp::message::ReaderOptions::new())
            .expect("read ForceInput");
    let input = message
        .get_root::<rgpot_core::Potentials_capnp::force_input::Reader>()
        .expect("ForceInput root");

    let got_positions = input.get_pos().expect("positions");
    let got_atomic_numbers = input.get_atmnrs().expect("atomic numbers");
    let got_box = input.get_box().expect("box");
    assert_eq!(
        (0..got_positions.len())
            .map(|i| got_positions.get(i))
            .collect::<Vec<_>>(),
        positions
    );
    assert_eq!(
        (0..got_atomic_numbers.len())
            .map(|i| got_atomic_numbers.get(i))
            .collect::<Vec<_>>(),
        atomic_numbers
    );
    assert_eq!(
        (0..got_box.len())
            .map(|i| got_box.get(i))
            .collect::<Vec<_>>(),
        box_matrix
    );
    assert_eq!(
        input.get_length_unit().unwrap().to_str().unwrap(),
        "angstrom"
    );
    assert_eq!(input.get_energy_unit().unwrap().to_str().unwrap(), "eV");
}

#[test]
fn force_input_codec_omits_cell_for_molecular_geometry() {
    let positions = [0.0, 0.0, 0.0, 0.9, 0.0, 0.0];
    let atomic_numbers = [8, 1];
    let encoded = encode_force_input(&ProfileRequest {
        positions: &positions,
        atomic_numbers: &atomic_numbers,
        box_matrix: None,
        length_unit: "angstrom",
        energy_unit: "eV",
    })
    .expect("encode molecular ForceInput");

    let mut bytes = encoded.as_slice();
    let message =
        serialize::read_message_from_flat_slice(&mut bytes, capnp::message::ReaderOptions::new())
            .expect("read ForceInput");
    let input = message
        .get_root::<rgpot_core::Potentials_capnp::force_input::Reader>()
        .expect("ForceInput root");

    assert_eq!(input.get_box().expect("box list").len(), 0);
}

#[test]
fn potential_result_codec_preserves_one_fused_evaluation() {
    let mut message = Builder::new_default();
    {
        let mut result = message.init_root::<potential_result::Builder>();
        result.set_energy(-12.75);
        let mut forces = result.reborrow().init_forces(6);
        for (i, value) in [1.0, 2.0, 3.0, -1.0, -2.0, -3.0].into_iter().enumerate() {
            forces.set(i as u32, value);
        }
    }
    let encoded = serialize::write_message_to_words(&message);
    let evaluation = decode_potential_result(&encoded, 6).expect("decode PotentialResult");

    assert_eq!(evaluation.energy, -12.75);
    assert_eq!(evaluation.forces, [1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
}
