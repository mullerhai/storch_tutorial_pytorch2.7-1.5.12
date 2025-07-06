import org.bytedeco.pytorch.{MNIST, Module, *}
import org.bytedeco.pytorch.global.torch.*


object SimpleMNIST {
// Define a new Module.
  class Nets extends Module {
// Construct and register two Linear submodules.
    final var  fc1 = register_module("fc1", new LinearImpl(784, 64))
    final var  fc2 = register_module("fc2", new LinearImpl(64, 32))
    final var  fc3 = register_module("fc3", new LinearImpl(32, 10))
// Implement the Net's algorithm.
    def forward(xs: Tensor): Tensor = {
// Use one of many tensor manipulation functions.
      var x = xs
      x = relu(fc1.forward(x.reshape(x.size(0), 784)))
      x = dropout(x, /*p=*/ 0.5, /*train=*/ is_training)
      x = relu(fc2.forward(x))
      x = log_softmax(fc3.forward(x), /*dim=*/ 1)
      x
    }
// Use one of many "standard library" modules.
//    final var fc1: LinearImpl = null
//    final var fc2: LinearImpl = null
//    final var fc3: LinearImpl = null
  }
  @throws[Exception]
  def main(args: Array[String]): Unit = {
    /* try to use MKL when available */
    System.setProperty("org.bytedeco.openblas.load", "mkl")
// Create a new Net.
    val net = new SimpleMNIST.Nets
    import org.bytedeco.pytorch.global.torch as torchNative
    println(torchNative.cuda_is_available())
    println(torchNative.cuda_device_count())


    // Create a multi-threaded data loader for the MNIST dataset. "D:\\\\data\\\\FashionMNIST"  "/Users/zhanghaining/Downloads/mnist"
    val data_set = new MNIST("D:\\\\data\\\\FashionMNIST").map(new ExampleStack)
    val data_loader = new MNISTRandomDataLoader(
      data_set,
      new RandomSampler(data_set.size.get),
      new DataLoaderOptions( /*batch_size=*/ 64)
    )
// Instantiate an SGD optimization algorithm to update our Net's parameters.
    val optimizer = new SGD(net.parameters, new SGDOptions( /*lr=*/ 0.01))
    for (epoch <- 1 to 10) {
      var batch_index = 0
// Iterate the data loader to yield batches from the dataset.
      var it = data_loader.begin
//      println(s" data loader batch index ${batch_index} it ${it}")
      while (!it.equals(data_loader.end)) {
        val batch = it.access()
//        println(s" batch ${batch}")
// Reset gradients.
        optimizer.zero_grad()
// Execute the model on the input data.
        val prediction = net.forward(batch.data)
// Compute a loss value to judge the prediction of our model.
        val loss = nll_loss(prediction, batch.target)
// Compute gradients of the loss w.r.t. the parameters of our model.
        loss.backward()
// Update the parameters based on the calculated gradients.
        optimizer.step
// Output the loss and checkpoint every 100 batches.
        if ({ batch_index += 1; batch_index } % 100 == 0) {
          System.out.println(
            "Epoch: " + epoch + " | Batch: " + batch_index + " | Loss: " + loss.item_float
          )
// Serialize your model periodically as a checkpoint.
          val archive = new OutputArchive
          net.save(archive)
          archive.save_to("net.pt")
        }

        it = it.increment()
      }
    }
  }
}
