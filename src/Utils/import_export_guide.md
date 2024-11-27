
## Tensor
Given a tensor of type T, where T is a zig type, its values will be reported in the following order and types:

- **usize** : `tensor.size`
- **usize** : shapeLenght, representing tensor.shape length
- **usize** : `tensor.shape[i]`for shapeLenght times
- **T** : `tensor.data[i]` for tensor.size times

## Layer
- **[10]u8** :  a `string` tag representing the type of layer  
Depending on the type of layer see the relative format. See Layer tags

### Activation Layer  

- **usize** : `n_inputs`
- **usize** : `n_neurons`
- **Tensor** : `input` tensor
- **Tensor** : `output` tensor
- **[10]u8** : `activationFunction`, see Activation Function tags

### Dense Layer
- **Tensor** : `weights` tensor
- **Tensor** : `bias` tensor
- **Tensor** : `input` tensor
- **Tensor** : `output` tensor
- **usize** : `n_inputs`
- **usize** : `n_neurons`
- **Tensor** : `w_gradients` tensor
- **Tensor** : `b_gradients` tensor

## Model
- **usize** : NoLayer, representing the number of layers in the model
- **Layer** : representing a layer. See Layer above.



