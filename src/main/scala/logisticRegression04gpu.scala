//import spire.random.Random

import basic.LstmNetApp.device
import basic.{LstmNet, NeuralNet}
import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.*
import torch.Device.{CPU, CUDA}
import torch.utils.data.dataset.ChunkSharedBatchDataset
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam
import java.nio.file.Paths
//import torchvision.datasets.FashionMNIST
import java.nio.file.Paths
//import scala.runtime.stdLibPatches.Predef.nn
import torch.internal.NativeConverters.{fromNative, toNative}
import scala.util.{Random, Using}

object lstmGpu extends App {

  @main
  def main(): Unit =
//    System.setProperty( "org.bytedeco.javacpp.logger.debug" , "true")
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val input_size = 28 * 28
    val hidden_size = 500
    val num_classes = 10
    val num_epochs = 500
    val batch_size = 100
    val learning_rate = 0.001f
    val dataPath = Paths.get("D:\\data\\FashionMNIST")
    val train_dataset = FashionMNIST(dataPath, train = true, download = true)
    val test_dataset = FashionMNIST(dataPath, train = false)
    println(s"torch.cuda.isAvailable ${torch.cuda.isAvailable}")
//    println(s"torch.cuda.device_count ${torch.cuda.}")
    val device = CUDA //if torch.cuda.isAvailable then CUDA else CPU
    val evalFeatures = test_dataset.features.to(device)
    val evalTargets = test_dataset.targets.to(device)
    val r = Random(seed = 0)

    def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
      r.shuffle(train_dataset).grouped(8).map { batch =>
        val (features, targets) = batch.unzip
        (torch.stack(features).to(device), torch.stack(targets).to(device))
      }

    val model = LstmNet().to(device)
//    val model = NeuralNet[Float32](input_size, hidden_size, num_classes).to(device)
//    val model =   nn.Linear(input_size, num_classes).to(device) //LstmNet().to(device) //
    val criterion = nn.loss.CrossEntropyLoss().to(device)
    val optimizer = torch.optim.SGD(model.parameters(true), lr = learning_rate)

    import org.bytedeco.pytorch.{ChunkDatasetOptions, DataLoaderOptions, Example, ExampleIterator, ExampleStack, ExampleVector, ExampleVectorIterator, JavaDataset, JavaDistributedRandomTensorDataLoader, JavaDistributedSequentialTensorDataLoader, JavaRandomDataLoader, JavaRandomTensorDataLoader, JavaSequentialTensorDataLoader, JavaStatefulDataset, JavaStreamDataLoader, RandomSampler, SizeTArrayRef, SizeTOptional, TensorExample, TensorExampleIterator, TensorExampleStack, TensorExampleVector, AbstractTensor as Tensor, ChunkDataReader as CDR, ChunkDataset as CD, ChunkRandomDataLoader as CRDL, ChunkSharedBatchDataset as CSBD, DistributedRandomSampler as DRS, DistributedSequentialSampler as DSS, JavaStreamDataset as JSD, JavaTensorDataset as TD, StreamSampler as STS}
//    import torch.utils.data.DataLoaderOptions
    import torch.utils.data.dataloader.*
    import torch.utils.data.datareader.{ChunkDataReader, ChunkTensorDataReader, ExampleVectorReader, TensorExampleVectorReader}
    import torch.utils.data.dataset.*
    import torch.utils.data.dataset.java.{StatefulDataset, StatefulTensorDataset, StreamDataset, StreamTensorDataset, TensorDataset, JavaDataset as JD}
    import torch.utils.data.sampler.{DistributedRandomSampler, DistributedSequentialSampler, StreamSampler, RandomSampler as RS, SequentialSampler as SS}

    def exampleVectorToExample(exVec: ExampleVector): Example = {
      val example = new Example(exVec.get(0).data(), exVec.get(0).target())
      example
    }


    val exampleSeq = train_dataset.map(x => new Example(x._1.native, x._2.native))
    //  val ex1 = new Example(mnistTrain.features.native ,mnistTrain.targets.native)
    val exampleVector = new ExampleVector(exampleSeq *)
    val reader = new ChunkDataReader()
    reader(exampleVector)
    val prefetch_count = 1
    //  val ds = new ChunkSharedBatchDataset(new ChunkDataset(reader, new RandomSampler(exampleSeq.size), new RandomSampler(exampleSeq.size), new ChunkDatasetOptions(prefetch_count, batch_size))).map(new ExampleStack)
    //  val ds  = new ChunkSharedTensorBatchDataset(new ChunkTensorDataset(reader,new RS(exampleTensorSeq.size),new ChunkDatasetOptions(prefetch_count, batch_size))).map(new TensorExampleStack)
    val ds = new ChunkSharedBatchDataset(
      new ChunkDataset(
        reader,
        new RandomSampler(exampleSeq.size),
        new RandomSampler(exampleSeq.size),
        new ChunkDatasetOptions(prefetch_count, batch_size)
      )
    ).map(new ExampleStack)

    val opts = new DataLoaderOptions(100)
    //  opts.workers.put(5)
    opts.batch_size.put(100)
    //  opts.enforce_ordering.put(true)
    //  opts.drop_last.put(false)
    val data_loader = new ChunkRandomDataLoader(ds, opts)
    val total_step = train_dataset.length // 2000 //data_loader //
    (1 to num_epochs).foreach(epoch => {
      var it: ExampleIterator = data_loader.begin
      var batchIndex = 0
      println("coming in for loop")
      while (!it.equals(data_loader.end)) {
        Using.resource(new PointerScope()) { p =>
          val batch = it.access
          optimizer.zeroGrad()
//          println(s"batch comming in ${batchIndex}")
//          val trainDataTensor = fromNative(batch.data()).to(device)
          val ze = batch.data().shape
          //          println(s"ze: ${ze(0)} 1: ${ze(1)} 2 : ${ze(2)} ${ze(3)}")
          val prediction = model(fromNative(batch.data().view(-1, 28 * 28)).reshape(-1, 28 , 28).to(device))
          val loss = criterion(prediction, fromNative(batch.target()).to(device))
          loss.backward()
          optimizer.step()
          it = it.increment
//          val accuracy =
//            (prediction.argmax(dim = 1).eq(fromNative(batch.target()).to(device)).sum / train_dataset.length).item
//
//          println(
//            f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Training loss: ${loss.item}%.4f | Eval accuracy: $accuracy%.4f"
//          )
//          println(s"train Loss grad_fn: ${loss.grad_fn()}")
//          println(s"train out grad_fn: ${prediction.grad_fn()}")
          batchIndex += 1
          if batchIndex % 200 == 0 then
            // run evaluation
            torch.noGrad {
              val correct = 0
              val total = 0
              println("coming eval...")
              val predictions = model(evalFeatures.reshape(-1, 28 ,28))
              println(s"predictions : ${predictions} \n")
              val evalLoss = criterion(predictions, evalTargets)
              println(s"evalLoss : ${evalLoss.item} \n")
//              val featuresData = new Array[Float](1000)
//              val fp4 = new FloatPointer(predictions.native.data_ptr_float())
//              fp4.get(featuresData)
//              println(s"\n ffff size ${featuresData.size} shape ${
//                evalFeatures.shape
//                  .mkString(", ")
//              }a data ${featuresData.mkString(" ")}")
              println(s"predictions : ${predictions} \n")
              //              println(s"loss grad_fn: ${evalLoss.grad_fn()}")
              val accuracy =
                (predictions.argmax(dim = 1).eq(evalTargets).sum / test_dataset.length).item
              println(
                f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
              )
            }

          //        it = it.increment

        }
      }

    })

    println(s"model ${model.modules.toSeq.mkString(" \n")}")
    println(s"model ${model.summarize}")

  //    Using.resource(torch.noGrad()){
  //      
  //    }
}
