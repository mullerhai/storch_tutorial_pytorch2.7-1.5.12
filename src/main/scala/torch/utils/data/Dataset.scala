//package torch
//package utils.data
//
//import basic.FashionMNIST
//import org.bytedeco.pytorch.{ChunkDatasetOptions, Example, ExampleIterator, ExampleStack, ExampleVector, ExampleVectorIterator, JavaDataset, JavaDistributedRandomTensorDataLoader, JavaDistributedSequentialTensorDataLoader, JavaRandomDataLoader, JavaRandomTensorDataLoader, JavaSequentialTensorDataLoader, JavaStatefulDataset, JavaStreamDataLoader, RandomSampler, SizeTArrayRef, SizeTOptional, TensorExample, TensorExampleIterator, TensorExampleStack, TensorExampleVector, AbstractTensor as RawTensor, ChunkDataReader as CDR, ChunkDataset as CD, ChunkRandomDataLoader as CRDL, ChunkSharedBatchDataset as CSBD, DataLoaderOptions as DLO, DistributedRandomSampler as DRS, DistributedSequentialSampler as DSS, JavaStreamDataset as JSD, JavaTensorDataset as TD, StreamSampler as STS}
//import torch.utils.data.DataLoaderOptions
//import torch.utils.data.dataloader.*
//import torch.utils.data.datareader.{ChunkDataReader, ChunkTensorDataReader, ExampleVectorReader, TensorExampleVectorReader}
//import torch.utils.data.dataset.*
//import torch.utils.data.dataset.java.{StatefulDataset, StatefulTensorDataset, StreamDataset, StreamTensorDataset, TensorDataset, JavaDataset as JD}
//import torch.utils.data.sampler.{DistributedRandomSampler, DistributedSequentialSampler, StreamSampler, RandomSampler as RS, SequentialSampler as SS}
//import java.nio.file.Paths
//
//
////trait Dataset[] {
////  
////  def init(data: Seq[Number]):Unit
////  
////  def len: Long
////  
////  def getItem(idx: Int): Tuple2[Tensor[ParamType],Tensor[ParamType]]
////}
//
////
////class Taobao2Dataset[ParamType <: DType :Default] extends Dataset[ParamType]{
////
////  override def init(data: Seq[Number]): Unit = ???
////
////  override def len: Long = ???
////
////  override def getItem(idx: Int): Tuple2[Tensor[ParamType],Tensor[ParamType]]= ???
////
////}
//
//
//object dataReader {
//
//  def main(args: Array[String]): Unit = {
//    val batch_size = 100
//    val dataPath = Paths.get("D:\\data\\FashionMNIST")
//    val train_dataset = FashionMNIST(dataPath, train = true, download = true)
//    val test_dataset = FashionMNIST(dataPath, train = false)
//    def exampleVectorToExample(exVec: ExampleVector): Example = {
//      val example = new Example(exVec.get(0).data(), exVec.get(0).target())
//      example
//    }
//
//
//    val exampleSeq = train_dataset.map(x => new Example(x._1.native, x._2.native))
//    //  val ex1 = new Example(mnistTrain.features.native ,mnistTrain.targets.native)
//    val exampleVector = new ExampleVector(exampleSeq *)
//    val reader = new ChunkDataReader()
//    reader(exampleVector)
//    val prefetch_count = 1
//    //  val ds = new ChunkSharedBatchDataset(new ChunkDataset(reader, new RandomSampler(exampleSeq.size), new RandomSampler(exampleSeq.size), new ChunkDatasetOptions(prefetch_count, batch_size))).map(new ExampleStack)
//    //  val ds  = new ChunkSharedTensorBatchDataset(new ChunkTensorDataset(reader,new RS(exampleTensorSeq.size),new ChunkDatasetOptions(prefetch_count, batch_size))).map(new TensorExampleStack)
//    val ds = new ChunkSharedBatchDataset(
//      new ChunkDataset(
//        reader,
//        new RandomSampler(exampleSeq.size),
//        new RandomSampler(exampleSeq.size),
//        new ChunkDatasetOptions(prefetch_count, batch_size)
//      )
//    ).map(new ExampleStack)
//
//    val opts = new DataLoaderOptions(100)
//    //  opts.workers.put(5)
//    opts.batch_size.put(100)
//    //  opts.enforce_ordering.put(true)
//    //  opts.drop_last.put(false)
//    val data_loader = new ChunkRandomDataLoader(ds, opts)
//    val total_step = train_dataset.length // 2000 //data_loader //
//  }
//}