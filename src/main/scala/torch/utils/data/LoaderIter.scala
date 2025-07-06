package torch
package utils.data

import org.bytedeco.pytorch.{ChunkDataset, ChunkDatasetOptions, ChunkMapDataset, ChunkRandomDataLoader, ChunkSharedBatchDataset, Example, ExampleIterator, ExampleStack, ExampleVector}
import torch.data.sampler.Sampler
import torch.data.DataLoaderOptions
import torch.DType
import torch.utils.data.Dataset
import torch.Tensor
import torch.data.datareader.ChunkDataReader

import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer

// 假设这些类型和特质已经定义
//trait DType
trait Dataset[ParamType <: DType : Default] {
  def init(data: AnyRef*): Unit
  
  def length: Long
  
  def getItem(idx: Int): (Tensor[ParamType], Tensor[ParamType])
  
  def apply(data:AnyRef*):Unit =
    init(data)
}

case class DataLoaderOptions(batch_size: Int = 1, shuffle: Boolean = false, sampler: Sampler = null,
                             batch_sampler: Sampler = null, num_workers: Int = 0, collate_fn: Any = null,
                             pin_memory: Boolean = false, drop_last: Boolean = false, timeout: Int = 0,
                             worker_init_fn: Any = null, prefetch_factor: Int = 2,
                             persistent_workers: Boolean = false)

//class Taobao3Dataset[ParamType <: DType : Default] extends Dataset[ParamType] {
//  override def init(data: Seq[Number]): Unit = ???
//  override def length: Long = ???
//  override def getItem(idx: Int): (Tensor[ParamType], Tensor[ParamType]) = ???
//}

// 定义一个可迭代的类，用于遍历用户自定义数据集
class DataLoaderIterable[ParamType <: DType : Default](dataset: Dataset[ParamType], options: DataLoaderOptions) extends Iterable[Example] {
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

object testLoader {
  
  def main(args: Array[String]): Unit = {
//    val dataset = new Taobao3Dataset[Float32]()
//    val loader = new DataLoaderIterable[Float32](dataset, DataLoaderOptions(batch_size = 10))
//    for (example <- loader) {
//      println(example)
//    }
  }
}