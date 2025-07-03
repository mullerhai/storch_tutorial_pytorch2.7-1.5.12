/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package nn
package modules

import torch.nn.modules.normalization.LocalResponseNorm
import torch.{Tensor, nn}

class LocalResponseNormSuite extends munit.FunSuite {
  test("LocalResponseNormSuite output shapes") {
    val m12 = new LocalResponseNorm(size = 2)
    val input1 = torch.randn(Seq(32, 5, 24, 24))
    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(m12(input1).shape, Seq(32, 5, 24, 24)) // torch.Size([100])
    //    println(m12(input))
  }
}

class RMSNormSuite extends munit.FunSuite {
  test("RMSNormSuite output shapes") {
    val m12 = nn.RMSNorm(normalized_shape = Seq(20, 30))
    val input1 = torch.randn(Seq(20, 20, 30))
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
//    assertEquals(m12(input1).shape, Seq(20, 6, 10, 10)) //torch.Size([20, 6, 10, 10])
    println(m12(input1))
  }
}

class GroupNormSuite extends munit.FunSuite {
  test("GroupNormSuite output shapes") {
    val m12 = nn.GroupNorm(num_groups = 3, num_channels = 6)
    val input1 = torch.randn(Seq(20, 6, 10, 10))
    //    val input2 = torch.randn(Seq(16, 5, 7, 7, 7, 7))
    assertEquals(m12(input1).shape, Seq(20, 6, 10, 10)) // torch.Size([20, 6, 10, 10])
    //    println(m12(input))
  }
}

class NormalizationSuite extends munit.FunSuite {

  test("LayerNorm") {
    {
      torch.manualSeed(0)
      val (batch, sentenceLength, embeddingDim) = (2, 2, 3)
      val embedding = torch.randn(Seq(batch, sentenceLength, embeddingDim))
      val layerNorm = nn.LayerNorm(Seq(embeddingDim))
      val output = layerNorm(embedding)
      assertEquals(output.shape, embedding.shape)
      val expectedOutput = Tensor(
        Seq(
          Seq(
            Seq(1.2191f, 0.0112f, -1.2303f),
            Seq(1.3985f, -0.5172f, -0.8813f)
          ),
          Seq(
            Seq(0.3495f, 1.0120f, -1.3615f),
            Seq(-0.3948f, -0.9786f, 1.3734f)
          )
        )
      )
      assert(torch.allclose(output, expectedOutput, atol = 1e-4))
    }
    {
      torch.manualSeed(0)
      val (n, c, h, w) = (1, 2, 2, 2)
      val input = torch.randn(Seq(n, c, h, w))
      // Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
      val layerNorm = nn.LayerNorm(Seq(c, h, w))
      val output = layerNorm(input)
      assertEquals(output.shape, (Seq(n, c, h, w)))
      val expectedOutput = Tensor(
        Seq(
          Seq(
            Seq(1.4715f, -0.0785f),
            Seq(-1.6714f, 0.6497f)
          ),
          Seq(
            Seq(-0.7469f, -1.0122f),
            Seq(0.5103f, 0.8775f)
          )
        )
      ).unsqueeze(0)
      assert(torch.allclose(output, expectedOutput, atol = 1e-4))
    }
  }

}
