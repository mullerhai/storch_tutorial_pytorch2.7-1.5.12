package torch
package utils.data

import basic.FashionMNIST
import org.bytedeco.pytorch.{ChunkDataset, ChunkDatasetOptions, ChunkMapDataset, ChunkRandomDataLoader, ChunkSharedBatchDataset, Example, ExampleIterator, ExampleStack, ExampleVector}

import scala.util.Random
import torch.data.sampler.{RandomSampler, Sampler}
import torch.data.DataLoaderOptions
import torch.DType
import torch.utils.data.Dataset
import torch.Tensor
import torch.data.datareader.ChunkDataReader
import torch.Device.{CPU, CUDA}
import org.bytedeco.javacpp.chrono.Milliseconds
import java.nio.file.Paths
import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer

// 假设这些类型和特质已经定义 , TargetType <: DType :Int64
//trait DType
trait Dataset[ParamType <: DType : Default ] {
//  def init(data: AnyRef*): Unit
  
  def length: Long
  
  def getItem(idx: Int): (Tensor[ParamType], Tensor[Int64])
  
//  def apply(data:AnyRef*):Unit =
//    init(data)
}

case class DataLoaderOptions(batch_size: Int = 1, shuffle: Boolean = true, sampler: Sampler = null,
                             batch_sampler: Sampler = null, num_workers: Int = 0, collate_fn: Any = null,
                             pin_memory: Boolean = false, drop_last: Boolean = false, timeout: Int = 0,
                             worker_init_fn: Any = null, prefetch_factor: Int = 2,
                             persistent_workers: Boolean = false)

class MnistDataset extends Dataset[Float32] {

  val device = if torch.cuda.isAvailable then CUDA else CPU

  println(s"Using device: $device")
  val dataPath = Paths.get("D:\\data\\FashionMNIST")
  val train_dataset = FashionMNIST(dataPath, train = true, download = true)
  val test_dataset = FashionMNIST(dataPath, train = false)
  val trainFeatures = train_dataset.features.to(device)
  val trainTargets = train_dataset.targets.to(device)
  val r = Random(seed = 0)

  def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
    r.shuffle(train_dataset).grouped(8).map { batch =>
      val (features, targets) = batch.unzip
      (torch.stack(features).to(device), torch.stack(targets).to(device))
    }
  val data = dataLoader
  println(s"train_dataset.features.shape.head ${train_dataset.features.shape.head}")
  override def length: Long = train_dataset.features.shape.head
  override def getItem(idx: Int): (Tensor[Float32], Tensor[Int64]) = {
    val feature: Tensor[Float32] = train_dataset.features(idx)
    val target: Tensor[Int64] = train_dataset.targets(idx)
    (feature,target)

  }
}

// 定义一个可迭代的类，用于遍历用户自定义数据集
class TorchDataLoader[ParamType <: DType : Default](dataset: Dataset[ParamType], options: DataLoaderOptions) extends Iterable[Example] {
  // 转换用户自定义数据集为 Example 序列
  private def convertDatasetToExamples(): Seq[Example] = {
    val examples = new ArrayBuffer[Example]()
    for (i <- 0 until dataset.length.toInt) {
      val (data, target) = dataset.getItem(i)
      // 这里需要根据实际的 Tensor 类型转换为 native 数据
      val example = new Example(data.native, target.native)
      examples += example
    }
    examples.toSeq
  }

  def exampleVectorToExample(exVec: ExampleVector): Example = {
    val example = new Example(exVec.get(0).data(), exVec.get(0).target())
    example
  }
  // 创建 ChunkDataReader
  private def createChunkDataReader(examples: Seq[Example]): ChunkDataReader = {
    val reader = new ChunkDataReader()
    val exampleVector = new org.bytedeco.pytorch.ExampleVector(examples*)
    reader(exampleVector)
    reader
  }

  // 创建 ChunkDataset
  private def createChunkDataset(reader: ChunkDataReader, examples: Seq[Example], options: DataLoaderOptions): ChunkDataset = {
    import torch.data.sampler.RandomSampler
    val prefetch_count = 1
    new ChunkDataset(
      reader,
      new RandomSampler(examples.size),
      new RandomSampler(examples.size),
      new ChunkDatasetOptions(prefetch_count, options.batch_size.toLong)
    )
  }

  // 创建 ChunkSharedBatchDataset
  private def createChunkSharedBatchDataset(chunkDataset: ChunkDataset): ChunkMapDataset = {
    new ChunkSharedBatchDataset(chunkDataset).map(new ExampleStack)
  }

  // 创建 ChunkRandomDataLoader
  private def createChunkRandomDataLoader(ds: ChunkMapDataset, options: DataLoaderOptions): ChunkRandomDataLoader = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(options.batch_size)
    loaderOpts.batch_size.put(options.batch_size)
//    loaderOpts.timeout().put(new Milliseconds(options.timeout.toLong))
    loaderOpts.drop_last().put(options.drop_last)
    loaderOpts.enforce_ordering().put(!options.shuffle)
    loaderOpts.workers().put(options.num_workers)
    loaderOpts.max_jobs().put(4)
    new ChunkRandomDataLoader(ds, loaderOpts)
  }

  // 初始化内部组件
  private val examples = convertDatasetToExamples()
  private val reader = createChunkDataReader(examples)
  private val nativeDataset: ChunkDataset = createChunkDataset(reader, examples, options)
  private val sharedBatchDataset = createChunkSharedBatchDataset(nativeDataset)
  private val nativeDataLoader: ChunkRandomDataLoader = createChunkRandomDataLoader(sharedBatchDataset, options)

  override def iterator: Iterator[Example] = new Iterator[Example] {
    private var current: ExampleIterator = nativeDataLoader.begin
    private val endIterator: ExampleIterator = nativeDataLoader.end

    // 检查是否还有下一个元素
    override def hasNext: Boolean = !current.equals(endIterator)

    // 获取下一个元素并移动迭代器
    override def next(): Example = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}

object datasetLoaderTraining {
  import torch.internal.NativeConverters.fromNative
  def main(args: Array[String]): Unit = {
    //    System.setProperty( "org.bytedeco.javacpp.logger.debug" , "true")
    System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
    val input_size = 28 * 28
    val hidden_size = 500
    val num_classes = 10
    val num_epochs = 50
    val batch_size = 100
    val learning_rate = 0.001f
    val inputDim = 28 * 28
    val dModel = 128
    val numExperts = 4
    val dFf = 512
    val numLayers = 2
    val numClasses = 10
    val dropout = 0.1
    val device = if torch.cuda.isAvailable then CUDA else CPU
    println(s"Using device: $device")
    val dataPath = Paths.get("D:\\data\\FashionMNIST")
    val test_dataset = FashionMNIST(dataPath, train = false)

    val evalFeatures = test_dataset.features.to(device)
    val evalTargets = test_dataset.targets.to(device)

    val dataset = new MnistDataset()
    val loader = new TorchDataLoader[Float32](dataset, DataLoaderOptions(batch_size = 60,shuffle = true , num_workers = 8))
    var index  = 0
    var batchIndex = 0
    val model = new MoETransformerClassifier[Float32](inputDim, dModel, numExperts, dFf, numLayers, numClasses, dropout).to(device)

    val criterion = nn.loss.CrossEntropyLoss().to(device)
    val optimizer = torch.optim.SGD(model.parameters(true), lr = learning_rate)
    (1 to num_epochs).foreach(epoch => {
      for (batch <- loader) {
        optimizer.zeroGrad()
        val prediction = model(fromNative(batch.data().view(-1, 28 * 28)).reshape(-1, 28 * 28).to(device))
        val loss = criterion(prediction, fromNative(batch.target()).to(device))
        loss.backward()
        optimizer.step()
        index += 1
        batchIndex += 1
        if batchIndex % 200 == 0 then
          // run evaluation
          torch.noGrad {
            val correct = 0
            val total = 0
            val predictions = model(evalFeatures.reshape(-1, 28 * 28))
            val evalLoss = criterion(predictions, evalTargets)
            println(s"evalLoss : ${evalLoss.item} \n")
            println(s"predictions : ${predictions} \n")

            val accuracy =
              (predictions.argmax(dim = 1).eq(evalTargets).sum / test_dataset.length).item
            println(
              f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
            )
          }
      }
    })

  }
}