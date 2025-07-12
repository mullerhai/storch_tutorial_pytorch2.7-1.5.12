//package torch.utils.data
//
//import org.bytedeco.pytorch.*
//import torch.utils.data.DataLoaderOptions
//import torch.utils.data.dataloader.ChunkRandomTensorDataLoader
//import torch.utils.data.datareader.ChunkTensorDataReader
//import torch.utils.data.sampler.Sampler
//import torch.utils.data.{Dataset, TensorDataLoaderOptions, TensorDataset}
//import torch.{DType, Default, Tensor}
//
//import scala.collection.Iterator
//import scala.collection.mutable.ArrayBuffer
//
//// 假设这些类型和特质已经定义
////trait DType  extends Dataset[ParamType]
//trait TensorDataset[ParamType <: DType : Default]  {
//  def init(data: AnyRef*): Unit
//  def len: Long
//  def getItem(idx: Int): Tensor[ParamType]
//
//  def apply(data: AnyRef*): Unit =
//    init(data)
//}
//
//case class TensorDataLoaderOptions(batch_size: Int = 1, shuffle: Boolean = false, sampler: Sampler = null,
//                             batch_sampler: Sampler = null, num_workers: Int = 0, collate_fn: Any = null,
//                             pin_memory: Boolean = false, drop_last: Boolean = false, timeout: Int = 0,
//                             worker_init_fn: Any = null, prefetch_factor: Int = 2,
//                             persistent_workers: Boolean = false)
//
//// 定义一个通用的 DataLoader trait
//trait DataLoader[ParamType <: DType : Default, ItemType] extends Iterable[ItemType] {
//  type DatasetType <: {
//    def len: Long
//    def getItem(idx: Int): Tensor[ParamType]
//  }
//  type OptionsType
//
//  val dataset: DatasetType
//  val options: OptionsType
//
//  // 转换用户自定义数据集为 ItemType 序列
//  private def convertDatasetToItems(): Seq[ItemType]
//
//  // 创建数据读取器
//  private def createDataReader(items: Seq[ItemType]): Any
//
//  // 创建数据集
//  private def createDataset(reader: Any, items: Seq[ItemType], options: OptionsType): Any
//
//  // 创建共享批次数据集
//  private def createSharedBatchDataset(dataset: Any): Any
//
//  // 创建数据加载器
//  private def createDataLoader(ds: Any, options: OptionsType): Any
//
//  // 初始化内部组件
//  private val items = convertDatasetToItems()
//  private val reader = createDataReader(items)
//  private val nativeDataset = createDataset(reader, items, options)
//  private val sharedBatchDataset = createSharedBatchDataset(nativeDataset)
//  private val nativeDataLoader = createDataLoader(sharedBatchDataset, options)
//
//  override def iterator: Iterator[ItemType]
//}
//
//// 定义一个可迭代的类，用于遍历用户自定义张量数据集
//class TorchTensorDataLoader[ParamType <: DType : Default](dataset: TensorDataset[ParamType], options: TensorDataLoaderOptions)
//  extends DataLoader[ParamType, TensorExample] {
//
//  override type DatasetType = TensorDataset[ParamType]
//  override type OptionsType = TensorDataLoaderOptions
//
//  override val dataset: TensorDataset[ParamType] = dataset
//  override val options: TensorDataLoaderOptions = options
//
//  // 转换用户自定义数据集为 TensorExample 序列
//  override private def convertDatasetToItems(): Seq[TensorExample] = {
//    val tensorExamples = new ArrayBuffer[TensorExample]()
//    for (i <- 0 until dataset.len.toInt) {
//      val data = dataset.getItem(i)
//      // 这里需要根据实际的 Tensor 类型转换为 native 数据
//      val tensorExample = new TensorExample(data.native)
//      tensorExamples += tensorExample
//    }
//    tensorExamples.toSeq
//  }
//
//  // 创建 TensorExampleVector
//  private def createTensorExampleVector(tensorExamples: Seq[TensorExample]): TensorExampleVector = {
//    new TensorExampleVector(tensorExamples*)
//  }
//
//  // 创建 ChunkTensorDataReader
//  override private def createDataReader(items: Seq[TensorExample]): ChunkTensorDataReader = {
//    val tensorExampleVector = createTensorExampleVector(items)
//    val reader = new ChunkTensorDataReader()
//    reader(tensorExampleVector)
//    reader
//  }
//
//  // 创建 ChunkTensorDataset
//  override private def createDataset(reader: ChunkTensorDataReader, items: Seq[TensorExample], options: TensorDataLoaderOptions): ChunkTensorDataset = {
//    import torch.utils.data.sampler.RandomSampler
//    val prefetch_count = 1
//    new ChunkTensorDataset(
//      reader,
//      new RandomSampler(items.size),
//      new RandomSampler(items.size),
//      new org.bytedeco.pytorch.ChunkDatasetOptions(prefetch_count, options.batch_size)
//    )
//  }
//
//  // 创建 ChunkSharedTensorBatchDataset
//  override private def createSharedBatchDataset(dataset: ChunkTensorDataset): ChunkMapTensorDataset = {
//    new ChunkSharedTensorBatchDataset(dataset).map(new TensorExampleStack)
//  }
//
//  // 创建 ChunkRandomTensorDataLoader（假设存在对应的类，根据实际情况调整）
//  override private def createDataLoader(ds: ChunkMapTensorDataset, options: TensorDataLoaderOptions): ChunkRandomTensorDataLoader = {
//    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(options.batch_size)
//    loaderOpts.batch_size.put(options.batch_size)
//    // 这里需要替换为实际的 ChunkRandomTensorDataLoader 构造函数
//    // 假设存在一个名为 ChunkRandomTensorDataLoader 的类
//    new ChunkRandomTensorDataLoader(ds, loaderOpts)
//  }
//
//  override def iterator: Iterator[TensorExample] = new Iterator[TensorExample] {
//    private var current: TensorExampleIterator = nativeDataLoader.begin.asInstanceOf[TensorExampleIterator]
//    private val endIterator: TensorExampleIterator = nativeDataLoader.end.asInstanceOf[TensorExampleIterator]
//
//    // 检查是否还有下一个元素
//    override def hasNext: Boolean = !current.equals(endIterator)
//
//    // 获取下一个元素并移动迭代器
//    override def next(): TensorExample = {
//      val batch = current.access
//      current = current.increment
//      batch
//    }
//  }
//}
