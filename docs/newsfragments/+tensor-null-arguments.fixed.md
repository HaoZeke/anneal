`rgpot_tensor_owned_cpu_f64_2d` returns NULL with an error message for NULL
data, a negative extent or an overflowing size, and `rgpot_tensor_device`,
`rgpot_tensor_data` and `rgpot_tensor_shape` return a safe value for a NULL
tensor. Before, these cases were undefined behaviour.
