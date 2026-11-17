// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#include "rgpot/MetatomicPot/vesin_compat.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include <ATen/CPUGeneratorImpl.h>
#include "vesin.h"

#include <torch/csrc/jit/runtime/graph_executor.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace std::string_literals;

namespace rgpot {

namespace {

// metatensor-torch 0.10.3 Module::to() first moves jit tensors, then walks
// every attribute when `_mts_buffer_names` is missing. Exported PET-MAD
// stores mixed dicts as ordinary attrs; empty containers count as
// non-metatensor and throw. Weights are already moved. Swallow only that
// mixed-dict error. Do not register `_mts_buffer_names` on scripted
// modules: register_attribute corrupts JIT object slots (ivalue getSlot).
bool is_mixed_mts_to_error(const c10::Error &e) {
  const std::string w = e.what_without_backtrace();
  return w.find("metatensor and non-metatensor") != std::string::npos;
}

void move_atomistic_model(metatensor_torch::Module &model, torch::Device device) {
  try {
    model.to(device);
  } catch (const c10::Error &e) {
    if (!is_mixed_mts_to_error(e)) {
      throw;
    }
  }
}

void move_atomistic_model(metatensor_torch::Module &model,
                          torch::ScalarType dtype) {
  try {
    model.to(dtype);
  } catch (const c10::Error &e) {
    if (!is_mixed_mts_to_error(e)) {
      throw;
    }
  }
}

// Haar-uniform SO(3) sample from a SEEDED CPU generator. The seed is the
// pass index, so every force call averages over the SAME orientation set:
// the averaged PES is then a fixed, smooth function of the positions.
// Drawing from the global RNG instead makes the effective PES stochastic
// call-to-call (same x, different E/F), which feeds a surrogate
// non-repeatable data and lets optimizer dynamics walk into clash-scale
// geometries.
torch::Tensor random_so3(torch::Device device, torch::ScalarType dtype,
                         uint64_t seed) {
  auto gen = at::detail::createCPUGenerator(seed);
  auto A = torch::randn({3, 3}, gen,
                        torch::TensorOptions().dtype(torch::kFloat64));
  auto qr = torch::linalg_qr(A);
  auto Q = std::get<0>(qr);
  auto R = std::get<1>(qr);
  auto d = torch::sign(torch::diagonal(R));
  Q = Q * d.unsqueeze(0);
  if (torch::det(Q).item<double>() < 0) {
    Q.select(1, 0).mul_(-1);
  }
  return Q.to(dtype).to(device);
}


// Rotation-group orbits as signed permutation matrices with det +1.
// Chiral octahedral O = all 24; tetrahedral T = the 12 with an even
// permutation part and an even number of sign flips. Averaging over a GROUP
// makes E_avg exactly G-invariant and F_avg = -grad E_avg an exact identity.
std::vector<torch::Tensor> rotation_group(long n, torch::Device device,
                                          torch::ScalarType dtype) {
  const int perms[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1},
                           {0, 2, 1}, {1, 0, 2}, {2, 1, 0}};
  const bool perm_even[6] = {true, true, true, false, false, false};
  std::vector<torch::Tensor> out;
  const bool want_tetra = n < 24;
  for (int p = 0; p < 6; ++p) {
    for (int sbits = 0; sbits < 8; ++sbits) {
      const int s0 = (sbits & 1) ? -1 : 1;
      const int s1 = (sbits & 2) ? -1 : 1;
      const int s2 = (sbits & 4) ? -1 : 1;
      const int n_minus = ((sbits & 1) != 0) + ((sbits & 2) != 0) +
                          ((sbits & 4) != 0);
      // det = sign(perm) * s0*s1*s2; keep proper rotations only.
      const int det = (perm_even[p] ? 1 : -1) * s0 * s1 * s2;
      if (det != 1) continue;
      if (want_tetra && (!perm_even[p] || (n_minus % 2) != 0)) continue;
      auto R = torch::zeros({3, 3},
                            torch::TensorOptions().dtype(torch::kFloat64));
      R[0][perms[p][0]] = static_cast<double>(s0);
      R[1][perms[p][1]] = static_cast<double>(s1);
      R[2][perms[p][2]] = static_cast<double>(s2);
      out.push_back(R.to(dtype).to(device));
    }
  }
  return out;
}

metatensor_torch::TensorMap scalar_system_data(const std::string &name,
                                               double value,
                                               torch::Device device,
                                               torch::ScalarType dtype) {
  auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto values = torch::tensor({value}, torch::TensorOptions().dtype(dtype).device(device))
                    .reshape({1, 1});
  auto samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
      "system", torch::zeros({1, 1}, int_opts));
  auto properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
      name, torch::zeros({1, 1}, int_opts));
  auto block = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
      values, samples, std::vector<metatensor_torch::Labels>{}, properties);
  auto keys = torch::make_intrusive<metatensor_torch::LabelsHolder>(
      "_", torch::zeros({1, 1}, int_opts));
  return torch::make_intrusive<metatensor_torch::TensorMapHolder>(
      keys, std::vector<metatensor_torch::TensorBlock>{block});
}

} // namespace

void apply_torch_determinism_policy(TorchDeterminismPolicy policy) {
  // Fast: leave process-global LibTorch flags alone so the default path keeps
  // fused CUDA attention and other high-throughput kernels available.
  if (policy != TorchDeterminismPolicy::Strict) {
    return;
  }

  // Strict: request deterministic algorithms (throw when an op has no
  // deterministic implementation) and pin scaled-dot-product attention to the
  // math SDP backend. Flash / memory-efficient / cuDNN SDP are fused CUDA
  // kernels with nondeterministic backward paths; math SDP remains enabled so
  // attention still runs. Also disable TF32 cuBLAS/cuDNN paths: on Ampere+
  // they quantize fp32 to a 10-bit mantissa, so the same model returns
  // different forces on CUDA vs CPU.
  //
  // Additional CUDA/cuDNN process flags that are NOT covered by
  // setDeterministicAlgorithms alone:
  //   - deterministicCuDNN: forces deterministic convolution algorithms
  //   - benchmarkCuDNN=false: auto-tuner picks different kernels run-to-run
  //   - deterministicFillUninitializedMemory: no garbage from uninitialized
  //     tensor storage under autograd
  //
  // CUBLAS_WORKSPACE_CONFIG is process *environment*, not at::globalContext.
  // For CUDA >= 10.2 bit-stable cuBLAS, the host must set
  // CUBLAS_WORKSPACE_CONFIG to ":4096:8" or ":16:8" *before* the first
  // cuBLAS call (see NVIDIA cuBLAS reproducibility docs and
  // at::Context::alertCuBLASConfigNotDeterministic). rgpot cannot inject that
  // into a parent process that already touched cuBLAS.
  //
  // These settings live on at::globalContext() and affect every Torch user
  // in the process.
  auto &ctx = at::globalContext();
  ctx.setDeterministicAlgorithms(true, /*warn_only=*/false);
  ctx.setDeterministicFillUninitializedMemory(true);
  ctx.setDeterministicCuDNN(true);
  ctx.setBenchmarkCuDNN(false);
  ctx.setSDPUseFlash(false);
  ctx.setSDPUseMemEfficient(false);
  ctx.setSDPUseCuDNN(false);
  ctx.setSDPUseMath(true);
  ctx.setAllowTF32CuBLAS(false);
  ctx.setAllowTF32CuDNN(false);
}

MetatomicPot::MetatomicPot(const MetatomicConfig &config)
    : Potential(PotType::Metatomic), m_config(config),
      m_model(torch::jit::Module()),
      m_device(torch::Device(c10::DeviceType::CPU)) {

  torch::jit::getProfilingMode() = false;

  // Optional process-global determinism (default Fast = no mutation).
  apply_torch_determinism_policy(m_config.torch_determinism);

  // 1. Load model
  torch::optional<std::string> extensions_directory = torch::nullopt;
  if (!m_config.extensions_directory.empty()) {
    extensions_directory = m_config.extensions_directory;
  }

  m_model = metatomic_torch::load_atomistic_model(m_config.model_path,
                                                  extensions_directory);

  // 2. Extract capabilities and neighbor list requests
  m_capabilities =
      m_model.run_method("capabilities")
          .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
  auto requests_ivalue = m_model.run_method("requested_neighbor_lists");
  for (const auto &request_ivalue : requests_ivalue.toList()) {
    auto request =
        request_ivalue.get()
            .toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
    m_nl_requests.push_back(request);
  }

  // The charge and multiplicity reach the model as the rgpot:: system data it
  // declares. With check_consistency metatomic rejects a System that lacks a
  // declared input or carries an undeclared one.
  if (m_model.find_method("requested_inputs")) {
    const auto inputs = m_model.run_method("requested_inputs").toGenericDict();
    m_attach_charge = inputs.contains(std::string("rgpot::charge"));
    m_attach_spin_multiplicity =
        inputs.contains(std::string("rgpot::spin_multiplicity"));
    m_attach_spin = inputs.contains(std::string("rgpot::spin"));
  }
  if (m_config.attach_system_extras) {
    m_attach_charge = m_attach_spin_multiplicity = m_attach_spin = true;
  }

  // 3. Device selection
  torch::optional<std::string> desired = torch::nullopt;
  if (!m_config.device.empty()) {
    desired = m_config.device;
  }
  auto device_type =
      metatomic_torch::pick_device(m_capabilities->supported_devices, desired);
  m_device = torch::Device(device_type);

  // 4. Data type
  if (m_capabilities->dtype() == "float64") {
    m_dtype = torch::kFloat64;
  } else if (m_capabilities->dtype() == "float32") {
    m_dtype = torch::kFloat32;
  } else {
    throw std::runtime_error("Unsupported dtype: " + m_capabilities->dtype());
  }
  if (!m_config.dtype_override.empty()) {
    if (m_config.dtype_override == "float64") {
      m_dtype = torch::kFloat64;
    } else if (m_config.dtype_override == "float32") {
      m_dtype = torch::kFloat32;
    }
  }

  move_atomistic_model(m_model, m_device);
  if (!m_config.dtype_override.empty() &&
      m_dtype != (m_capabilities->dtype() == "float64" ? torch::kFloat64
                                                       : torch::kFloat32)) {
    try {
      move_atomistic_model(m_model, m_dtype);
    } catch (const std::exception &) {
      if (m_capabilities->dtype() == "float64") {
        m_dtype = torch::kFloat64;
      } else {
        m_dtype = torch::kFloat32;
      }
    }
  }

  // 5. Resolve energy output key
  auto outputs = m_capabilities->outputs();
  m_energy_key =
      metatomic_torch::pick_output("energy", outputs, torch::nullopt);
  if (!outputs.contains(m_energy_key)) {
    throw std::runtime_error("Missing energy output in metatomic model");
  }

  // 6. Evaluation options
  m_eval_options =
      torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
  m_eval_options->set_length_unit(m_config.length_unit);

  auto model_output = outputs.at(m_energy_key);
  auto requested_output =
      torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
  requested_output->per_atom = model_output->per_atom;
  requested_output->explicit_gradients = {};
  requested_output->set_quantity("energy");
  requested_output->set_unit("eV");
  m_eval_options->outputs.insert(m_energy_key, requested_output);

  // Optional per-atom energy_uncertainty (eOn metatomic UQ path)
  m_uncertainty_threshold = m_config.uncertainty_threshold;
  if (m_uncertainty_threshold > 0) {
    try {
      m_energy_uncertainty_key = metatomic_torch::pick_output(
          "energy_uncertainty", outputs, torch::nullopt);
      if (outputs.contains(m_energy_uncertainty_key)) {
        auto uq_info = outputs.at(m_energy_uncertainty_key);
        if (uq_info->per_atom) {
          auto requested_uq =
              torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
          requested_uq->per_atom = true;
          requested_uq->explicit_gradients = {};
          requested_uq->set_quantity("energy");
          requested_uq->set_unit("eV");
          m_eval_options->outputs.insert(m_energy_uncertainty_key, requested_uq);
        } else {
          m_uncertainty_threshold = -1.0;
        }
      } else {
        m_uncertainty_threshold = -1.0;
      }
    } catch (...) {
      m_uncertainty_threshold = -1.0;
    }
  }

  m_check_consistency = m_config.check_consistency;
}

void MetatomicPot::attachSystemExtras(
    const metatomic_torch::System &system) const {
  // add_data rejects bare names that are not known outputs; extras must be
  // <domain>::<name>. The multiplicity goes under the metatomic name and the
  // UMA/fairchem "spin" alias.
  const auto charge = static_cast<double>(m_config.charge);
  const auto spin = static_cast<double>(m_config.spin);
  if (m_attach_charge) {
    system->add_data("rgpot::charge",
                     scalar_system_data("charge", charge, m_device, m_dtype));
  }
  if (m_attach_spin_multiplicity) {
    system->add_data(
        "rgpot::spin_multiplicity",
        scalar_system_data("spin_multiplicity", spin, m_device, m_dtype));
  }
  if (m_attach_spin) {
    system->add_data("rgpot::spin",
                     scalar_system_data("spin", spin, m_device, m_dtype));
  }
}



void MetatomicPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::lock_guard<std::mutex> lock(m_mutex);

  long nAtoms = static_cast<long>(in.nAtoms);
  const bool use_rotation =
      m_config.random_rotation || m_config.n_symmetry_rotations > 0;
  const bool probe_scatter =
      m_config.so3_probe_scatter && m_config.n_symmetry_rotations > 0;
  // Group orbit when the request reaches a group size; identity-first probe
  // mode adds the unrotated pass in front of the probes.
  std::vector<torch::Tensor> group_rots;
  if (m_config.n_symmetry_rotations >= 12) {
    group_rots = rotation_group(m_config.n_symmetry_rotations, m_device,
                                m_dtype);
  }
  const long n_rot_passes =
      !group_rots.empty() ? static_cast<long>(group_rots.size())
                          : (m_config.n_symmetry_rotations > 0
                                 ? m_config.n_symmetry_rotations
                                 : 1);
  const long n_passes = probe_scatter ? n_rot_passes + 1 : n_rot_passes;

  auto f64_options =
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

  if (m_cached_natoms != in.nAtoms) {
    std::vector<int32_t> types_vec(in.atmnrs, in.atmnrs + nAtoms);
    m_cached_types =
        torch::tensor(types_vec, torch::TensorOptions().dtype(torch::kInt32))
            .to(m_device);
    m_cached_natoms = in.nAtoms;
  }
  auto atomic_types = m_cached_types;

  double energy_acc = 0.0;
  auto forces_acc = torch::zeros({nAtoms, 3}, f64_options);
  // Force samples (CPU float64, n_passes x nAtoms x 3) for orientation force
  // uncertainty in eV/A — energy scatter is not a force-tolerance scale.
  std::vector<torch::Tensor> force_samples;
  if (n_passes > 1) {
    force_samples.reserve(static_cast<size_t>(n_passes));
  }
  double model_uq_variance = 0.0;
  bool have_model_uq = false;

  for (long i_pass = 0; i_pass < n_passes; ++i_pass) {
    torch::Tensor R = torch::eye(3, torch::TensorOptions()
                                        .dtype(m_dtype)
                                        .device(m_device));
    const bool identity_pass = probe_scatter && i_pass == 0;
    if (use_rotation && !identity_pass) {
      const long i_rot = probe_scatter ? i_pass - 1 : i_pass;
      if (!group_rots.empty()) {
        R = group_rots[static_cast<size_t>(i_rot)];
      } else {
        // Fixed per-pass seed: pass i uses the same orientation on every
        // call, so the averaged PES is deterministic (Monte Carlo regime,
        // n_symmetry_rotations < 12).
        R = random_so3(m_device, m_dtype,
                       0x50333A5EEDULL + static_cast<uint64_t>(i_rot));
      }
    }
    auto R_cpu = R.to(torch::kCPU).to(torch::kFloat64);

    auto pos_cpu = torch::from_blob(const_cast<double *>(in.pos),
                                    {nAtoms, 3}, f64_options)
                       .clone();
    auto cell_cpu =
        torch::from_blob(const_cast<double *>(in.box), {3, 3}, f64_options)
            .clone();
    if (use_rotation && !identity_pass) {
      // Rotate about the centroid: rotation about the origin also drags the
      // molecule through the model's input space, mixing orientation error
      // with translation error.
      auto centroid = pos_cpu.mean(0, /*keepdim=*/true);
      pos_cpu = (pos_cpu - centroid).matmul(R_cpu.transpose(0, 1)) + centroid;
      cell_cpu = cell_cpu.matmul(R_cpu.transpose(0, 1));
    }

    std::vector<double> pos_buf(static_cast<size_t>(nAtoms) * 3);
    std::vector<double> cell_buf(9);
    std::memcpy(pos_buf.data(), pos_cpu.contiguous().data_ptr<double>(),
                pos_buf.size() * sizeof(double));
    std::memcpy(cell_buf.data(), cell_cpu.contiguous().data_ptr<double>(),
                9 * sizeof(double));

    auto torch_positions =
        torch::from_blob(pos_buf.data(), {nAtoms, 3}, f64_options)
            .to(m_dtype)
            .to(m_device)
            .set_requires_grad(true);

    auto torch_cell =
        torch::from_blob(cell_buf.data(), {3, 3}, f64_options)
            .to(m_dtype)
            .to(m_device);

    auto cell_norms = torch::norm(torch_cell, 2, /*dim=*/1);
    auto torch_pbc = cell_norms.abs() > 1e-9;
    bool periodic[3] = {torch_pbc[0].item<bool>(), torch_pbc[1].item<bool>(),
                        torch_pbc[2].item<bool>()};

    auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
        atomic_types, torch_positions, torch_cell, torch_pbc);
    attachSystemExtras(system);

    for (const auto &request : m_nl_requests) {
      auto neighbors = computeNeighbors(request, nAtoms, pos_buf.data(),
                                        cell_buf.data(), periodic);
      metatomic_torch::register_autograd_neighbors(system, neighbors,
                                                   m_check_consistency);
      system->add_neighbor_list(request, neighbors);
    }

    auto ivalue_output = m_model.forward({
        std::vector<metatomic_torch::System>{system},
        m_eval_options,
        m_check_consistency,
    });
    auto dict_output = ivalue_output.toGenericDict();
    auto output_map = dict_output.at(m_energy_key)
                          .toCustomClass<metatensor_torch::TensorMapHolder>();

    if (m_uncertainty_threshold > 0 && i_pass == 0 &&
        !m_energy_uncertainty_key.empty() &&
        dict_output.contains(m_energy_uncertainty_key)) {
      try {
        auto uncertainty_map =
            dict_output.at(m_energy_uncertainty_key)
                .toCustomClass<metatensor_torch::TensorMapHolder>();
        auto uncertainty_block =
            metatensor_torch::TensorMapHolder::block_by_id(uncertainty_map, 0);
        auto flat_uncertainty =
            uncertainty_block->values().reshape({-1}).to(torch::kCPU);
        if (flat_uncertainty.numel() > 0) {
          model_uq_variance =
              flat_uncertainty.to(torch::kFloat64).mean().item<double>();
          have_model_uq = true;
        }
      } catch (...) {
      }
    }

    auto energy_block =
        metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
    auto energy_tensor = energy_block->values();
    const double e_pass = energy_tensor.sum().item<double>();
    if (!probe_scatter) {
      energy_acc += e_pass;
    }

    energy_tensor.backward(torch::ones_like(energy_tensor));
    auto positions_grad = system->positions().grad();
    auto forces_tensor =
        (-positions_grad).to(torch::kCPU).to(torch::kFloat64);
    if (use_rotation) {
      forces_tensor = forces_tensor.matmul(R_cpu);
    }
    if (probe_scatter && identity_pass) {
      // Probe mode: the unrotated pass alone steers geometry (energy and
      // forces from this pass only).
      energy_acc = e_pass;
      forces_acc = forces_tensor.clone();
    } else if (probe_scatter) {
      force_samples.push_back(forces_tensor.clone());
    } else {
      forces_acc += forces_tensor;
      if (n_passes > 1) {
        force_samples.push_back(forces_tensor.clone());
      }
    }
  }

  const double inv_n =
      probe_scatter ? 1.0 : 1.0 / static_cast<double>(n_passes);
  out->energy = energy_acc * inv_n;
  forces_acc = forces_acc * inv_n;
  std::memcpy(out->F, forces_acc.contiguous().data_ptr<double>(),
              in.nAtoms * 3 * sizeof(double));

  if (have_model_uq) {
    // Model energy_uncertainty mean (eV); callers may use as force floor at
    // unit length (angstrom) — that is an explicit energy->force scale choice.
    out->variance = model_uq_variance;
  } else if (n_passes > 1 && !force_samples.empty()) {
    // RMS force residual over orientations (eV/A): proper force-scale
    // uncertainty for criterion smearing. Energy orientation scatter can be
    // several eV on non-invariant evaluations and must not raise force tol.
    double acc2 = 0.0;
    const double n_comp = static_cast<double>(nAtoms) * 3.0 *
                          static_cast<double>(force_samples.size());
    // Deviation reference: the output force -- the orientation-average in
    // averaging mode, the unrotated force in probe mode. Either way the
    // RMS measures how much orientation moves the force actually used.
    for (const auto &F_i : force_samples) {
      auto d = F_i - forces_acc;
      acc2 += d.square().sum().item<double>();
    }
    out->variance = (n_comp > 0.0) ? std::sqrt(acc2 / n_comp) : 0.0;
  } else {
    out->variance = 0.0;
  }
}


void MetatomicPot::forceBatchImpl(const ForceBatch &batch) const {
  const bool plain =
      !m_config.random_rotation && m_config.n_symmetry_rotations <= 0;
  auto sequential = [&]() {
    for (size_t i = 0; i < batch.nSystems; ++i) {
      forceImpl(batch.in[i], &batch.out[i]);
    }
  };
  if (!plain || batch.nSystems < 2) {
    sequential();
    return;
  }

  try {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto f64_options =
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    std::vector<metatomic_torch::System> systems;
    std::vector<torch::Tensor> pos_tensors;
    systems.reserve(batch.nSystems);
    pos_tensors.reserve(batch.nSystems);

    for (size_t s = 0; s < batch.nSystems; ++s) {
      const ForceInput &in = batch.in[s];
      const long nAtoms = static_cast<long>(in.nAtoms);
      if (in.atmnrs == nullptr) {
        throw std::runtime_error("metatomic batch: atomic numbers are required");
      }
      auto torch_positions =
          torch::from_blob(const_cast<double *>(in.pos), {nAtoms, 3}, f64_options)
              .to(m_dtype)
              .to(m_device)
              .set_requires_grad(true);
      pos_tensors.push_back(torch_positions);
      auto torch_cell =
          torch::from_blob(const_cast<double *>(in.box), {3, 3}, f64_options)
              .to(m_dtype)
              .to(m_device);
      auto cell_norms = torch::norm(torch_cell, 2, /*dim=*/1);
      auto torch_pbc = cell_norms.abs() > 1e-9;
      bool periodic[3] = {torch_pbc[0].item<bool>(), torch_pbc[1].item<bool>(),
                          torch_pbc[2].item<bool>()};
      std::vector<int32_t> types_vec(in.atmnrs, in.atmnrs + nAtoms);
      auto atomic_types =
          torch::tensor(types_vec, torch::TensorOptions().dtype(torch::kInt32))
              .to(m_device);
      auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
          atomic_types, torch_positions, torch_cell, torch_pbc);
      attachSystemExtras(system);
      for (const auto &request : m_nl_requests) {
        auto neighbors = computeNeighbors(request, nAtoms, in.pos, in.box,
                                          periodic);
        metatomic_torch::register_autograd_neighbors(system, neighbors,
                                                     m_check_consistency);
        system->add_neighbor_list(request, neighbors);
      }
      systems.push_back(system);
    }

    auto ivalue_output = m_model.forward({
        systems,
        m_eval_options,
        m_check_consistency,
    });
    auto dict_output = ivalue_output.toGenericDict();
    auto output_map = dict_output.at(m_energy_key)
                          .toCustomClass<metatensor_torch::TensorMapHolder>();
    auto energy_block =
        metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
    auto energy_values = energy_block->values();
    auto samples = energy_block->samples();
    const bool per_atom = samples->size() > 0 && samples->names().size() > 1;
    if (per_atom) {
      auto system_col = samples->column("system").to(torch::kCPU);
      auto flat_energies = energy_values.reshape({-1}).to(torch::kCPU);
      for (size_t s = 0; s < batch.nSystems; ++s) {
        auto mask = (system_col == static_cast<int64_t>(s));
        batch.out[s].energy = flat_energies.index({mask})
                                  .sum()
                                  .to(torch::kFloat64)
                                  .item<double>();
      }
      energy_values.sum().backward();
    } else {
      auto cpu_energies = energy_values.to(torch::kCPU).to(torch::kFloat64);
      for (size_t s = 0; s < batch.nSystems; ++s) {
        batch.out[s].energy = cpu_energies[static_cast<int64_t>(s)].sum().item<double>();
      }
      energy_values.backward(torch::ones_like(energy_values));
    }
    for (size_t s = 0; s < batch.nSystems; ++s) {
      auto positions_grad = pos_tensors[s].grad();
      if (!positions_grad.defined()) {
        throw std::runtime_error("metatomic batch: a system produced no force");
      }
      auto forces_tensor = (-positions_grad).to(torch::kCPU).to(torch::kFloat64);
      const size_t n = batch.in[s].nAtoms * 3;
      std::memcpy(batch.out[s].F, forces_tensor.contiguous().data_ptr<double>(),
                  n * sizeof(double));
      batch.out[s].variance = 0.0;
      batch.out[s].has_stress = 0;
    }
  } catch (const std::exception &) {
    sequential();
  }
}

metatensor_torch::TensorBlock MetatomicPot::computeNeighbors(
    metatomic_torch::NeighborListOptions request, long nAtoms,
    const double *positions, const double *box, const bool periodic[3]) const {

  auto cutoff = request->engine_cutoff(m_config.length_unit);
  nlist::Query query;
  query.cutoff = cutoff;
  query.shape = request->full_list() ? nlist::Shape{nlist::Pairs{nlist::Shell::Full}}
                                     : nlist::Shape{nlist::Pairs{nlist::Shell::Half}};
  query.cut = nlist::Cut::Strict;
  if constexpr (requires { request->strict(); }) {
    query.cut = request->strict() ? nlist::Cut::Strict : nlist::Cut::Loose;
  }
  query.periodic = {periodic[0], periodic[1], periodic[2]};
  query.fields.shifts = true;
  query.fields.vectors = true;
  query.fields.distances = false;
  auto held = std::make_shared<nlist::Table>();
  nlist::build(*held, positions, static_cast<std::size_t>(nAtoms), box, query);

  auto n_pairs = static_cast<int64_t>(held->size());
  auto labels_options_cpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto pair_samples_values = torch::empty({n_pairs, 5}, labels_options_cpu);
  if (n_pairs > 0) {
    auto pair_samples_values_ptr = pair_samples_values.accessor<int32_t, 2>();
    for (int64_t i = 0; i < n_pairs; i++) {
      const auto p = static_cast<std::size_t>(i);
      pair_samples_values_ptr[i][0] = held->first[p];
      pair_samples_values_ptr[i][1] = held->second[p];
      pair_samples_values_ptr[i][2] = held->shift[3 * p];
      pair_samples_values_ptr[i][3] = held->shift[3 * p + 1];
      pair_samples_values_ptr[i][4] = held->shift[3 * p + 2];
    }
  }

  // The distance vectors stay in the table. The extra axis is a view.
  auto pair_vectors =
      n_pairs == 0
          ? torch::empty({0, 3, 1}, torch::TensorOptions().dtype(torch::kFloat64))
          : torch::from_blob(held->vector.data(), {n_pairs, 3},
                             [held](void *) {},
                             torch::TensorOptions().dtype(torch::kFloat64))
                .unsqueeze(-1);

  auto neighbor_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
      std::vector<std::string>{"first_atom", "second_atom", "cell_shift_a",
                               "cell_shift_b", "cell_shift_c"},
      pair_samples_values.to(m_device));

  auto labels_options_dev =
      torch::TensorOptions().dtype(torch::kInt32).device(m_device);
  auto neighbor_component =
      torch::make_intrusive<metatensor_torch::LabelsHolder>(
          "xyz", torch::tensor({0, 1, 2}, labels_options_dev).reshape({3, 1}));
  auto neighbor_properties =
      torch::make_intrusive<metatensor_torch::LabelsHolder>(
          "distance", torch::zeros({1, 1}, labels_options_dev));

  return torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
      pair_vectors.to(m_dtype).to(m_device), neighbor_samples,
      std::vector<metatensor_torch::Labels>{neighbor_component},
      neighbor_properties);
}

} // namespace rgpot
