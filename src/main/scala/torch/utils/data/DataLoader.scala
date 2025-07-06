package torch
package utils.data

import org.bytedeco.pytorch.ExampleIterator
import torch.data.dataloader.ChunkRandomDataLoader
import torch.data.dataset.ChunkDataset
import torch.data.sampler.Sampler

trait DataLoader[ParamType <: DType :Default](dataset: Dataset[ParamType], options: DataLoaderOptions ) extends Iterable[ParamType] {

  val nativeDataLoader: ChunkRandomDataLoader
  val nativeDataset: ChunkDataset
  def begin(): ExampleIterator

  def end(): ExampleIterator

  def join(): Unit

  def next(): ExampleIterator


  override def iterator: Iterator[ParamType] = ???
}

class ChunkDataLoader[ParamType <: DType : Default](dataset: Dataset[ParamType], options: DataLoaderOptions) extends DataLoader[ParamType](dataset, options)  {

  override val nativeDataLoader: ChunkRandomDataLoader = ???

  override val nativeDataset: ChunkDataset = ???
  
  def begin(): ExampleIterator

  def end(): ExampleIterator

  def join(): Unit

  def next(): ExampleIterator


}


case class DataLoaderOptions(batch_size: Int =1, shuffle: Boolean =false, sampler: Sampler=None,
                             batch_sampler: Sampler=None, num_workers: Int =0, collate_fn=None,
                             pin_memory: Boolean =false, drop_last: Boolean = false, timeout : Int =0,
                             worker_init_fn=None, prefetch_factor: Int=2,
                             persistent_workers: Boolean =false)