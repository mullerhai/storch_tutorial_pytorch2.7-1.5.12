//package torch.utils.data
//
//import org.bytedeco.pytorch.{DataLoaderOptions, ExampleVector, ExampleVectorIterator, ExampleVectorReader, JavaStreamDataLoader, StreamBatchDataset, StreamDataset, StreamSampler}
//import torch.data.datareader.ExampleVectorReader
//import torch.data.dataset.java.{StreamBatchDataset, StreamDataset}
//
//import scala.collection.Iterator
//import scala.collection.mutable.ArrayBuffer
//
//// 定义一个函数类型，用于从 Spark 或 Flink 获取 ExampleVector 数据
//type StreamingDataFetcher = () => Iterator[ExampleVector]
//
//class StreamDataLoader(
//    streamingDataFetcher: StreamingDataFetcher,
//    sampler: StreamSampler,
//    options: DataLoaderOptions
//) extends Iterable[ExampleVector] {
//
//  // 创建 ExampleVectorReader
//  private def createExampleVectorReader(): ExampleVectorReader = {
//    new ExampleVectorReader()
//  }
//
//  // 创建 StreamDataset，将流式数据转换为 StreamDataset
//  private def createStreamDataset(reader: ExampleVectorReader): StreamDataset = {
//    val dataIterator = streamingDataFetcher()
//    val exampleVectors = new ArrayBuffer[ExampleVector]()
//    dataIterator.foreach(exampleVectors += _)
//    new StreamDataset(reader, exampleVectors.toArray)
//  }
//
//  // 创建 StreamBatchDataset
//  private def createStreamBatchDataset(dataset: StreamDataset): StreamBatchDataset = {
//    new StreamBatchDataset(dataset, sampler)
//  }
//
//  // 创建 JavaStreamDataLoader
//  private def createJavaStreamDataLoader(dataset: StreamBatchDataset): JavaStreamDataLoader = {
//    new JavaStreamDataLoader(dataset, sampler, options)
//  }
//
//  // 初始化内部组件
//  private val reader: ExampleVectorReader = createExampleVectorReader()
//  private val streamDataset: StreamDataset = createStreamDataset(reader)
//  private val streamBatchDataset: StreamBatchDataset = createStreamBatchDataset(streamDataset)
//  private val javaStreamDataLoader: JavaStreamDataLoader = createJavaStreamDataLoader(streamBatchDataset)
//
//  override def iterator: Iterator[ExampleVector] = new Iterator[ExampleVector] {
//    private var current: ExampleVectorIterator = javaStreamDataLoader.begin()
//    private val endIterator: ExampleVectorIterator = javaStreamDataLoader.end()
//
//    // 检查是否还有下一个元素
//    override def hasNext: Boolean = !current.equals(endIterator)
//
//    // 获取下一个元素并移动迭代器
//    override def next(): ExampleVector = {
//      val batch = current.access()
//      current = current.increment()
//      batch
//    }
//  }
//}
//
//// 模拟从 Spark 或 Flink 获取数据的函数
//def mockStreamingDataFetcher(): Iterator[ExampleVector] = {
//  // 这里模拟返回一些 ExampleVector 数据
//  val exampleVectors = Seq.fill(10)(new ExampleVector())
//  exampleVectors.iterator
//}
//
//// 创建 StreamSampler 和 DataLoaderOptions
//import torch.data.sampler.StreamSampler
//import torch.data.DataLoaderOptions
//
//val sampler = new StreamSampler()
//val options = new DataLoaderOptions(batch_size = 2)
//
//// 创建 StreamDataLoader 实例
//val streamDataLoader = new StreamDataLoader(mockStreamingDataFetcher, sampler, options)
