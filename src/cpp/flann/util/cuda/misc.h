#ifndef FLANN_UTIL_CUDA_MISC_H
#define FLANN_UTIL_CUDA_MISC_H

/*
    Copyright (c) 2011, Andreas Mützel <andreas.muetzel@gmx.net>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
 * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Andreas Mützel <andreas.muetzel@gmx.net> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Andreas Mützel <andreas.muetzel@gmx.net> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace flann
{
namespace cuda
{

template <typename keyT, typename valueT, int size>
__device__ __forceinline__ void 
warp_reduce_min ( volatile keyT* smem, volatile valueT* value )
{
    if (size >=64 )
    {
        if ( smem[threadIdx.x+32] < smem[threadIdx.x] )
        {
            smem[threadIdx.x] = smem[threadIdx.x+32];
            value[threadIdx.x] = value[threadIdx.x+32];
        }
    }
    if (size >=32 )
    {
        if ( smem[threadIdx.x+16] < smem[threadIdx.x] )
        {
            smem[threadIdx.x] = smem[threadIdx.x+16];
            value[threadIdx.x] = value[threadIdx.x+16];
        }
    }
    if (size >=16 )
    {
        if ( smem[threadIdx.x+8] < smem[threadIdx.x] )
        {
            smem[threadIdx.x] = smem[threadIdx.x+8];
            value[threadIdx.x] = value[threadIdx.x+8];
        }
    }
    if (size >=8 )
    {
        if ( smem[threadIdx.x+4] < smem[threadIdx.x] )
        {
            smem[threadIdx.x] = smem[threadIdx.x+4];
            value[threadIdx.x] = value[threadIdx.x+4];
        }
    }
    if (size >=4 )
    {
        if ( smem[threadIdx.x+2] < smem[threadIdx.x] )
        {
            smem[threadIdx.x] = smem[threadIdx.x+2];
            value[threadIdx.x] = value[threadIdx.x+2];
        }
    }
    if (size >=2 )
    {
        if ( smem[threadIdx.x+1] < smem[threadIdx.x] )
        {
            smem[threadIdx.x] = smem[threadIdx.x+1];
            value[threadIdx.x] = value[threadIdx.x+1];
        }
    }
}

template <typename keyT, int size>
__device__ __forceinline__ void 
warp_reduce_sum ( volatile keyT* smem )
{
    if (size >=64 )
    {
        smem[threadIdx.x] += smem[threadIdx.x+32];
    }
    if (size >=32 )
    {
        smem[threadIdx.x] += smem[threadIdx.x+16];
    }
    if (size >=16 )
    {
        smem[threadIdx.x] += smem[threadIdx.x+8];
    }
    if (size >=8 )
    {
        smem[threadIdx.x] += smem[threadIdx.x+4];
    }
    if (size >=4 )
    {
        smem[threadIdx.x] += smem[threadIdx.x+2];
    }
    if (size >=2 )
    {
        smem[threadIdx.x] += smem[threadIdx.x+1];
    }
}


//! sorts the 32 elements (key-value-pairs)  
//! adapted from http://www.bealto.com/gpu-sorting_parallel-merge-local.html, thanks to Eric Bainville for the code!
template< typename Key, typename Value, bool ascending >
__device__ void 
sort_warp( Key* key, Value *value, unsigned len = 32 )
{
  int i = threadIdx.x;
  // Now we will merge sub-sequences of length 1,2,...,WG/2
  for (int length=1;length<len;length<<=1)
  {
    int iData = value[i];
    float iKey = key[i];
    int ii = i & (length-1);  // index in our sequence in 0..length-1
    int sibling = (i - ii) ^ length; // beginning of the sibling sequence
    int pos = 0;
    for (int inc=length;inc>0;inc>>=1) // increment for dichotomic search
    {
      int j = sibling+pos+inc-1;
      float jKey = key[j];
      bool smaller = ( ascending? jKey < iKey : jKey > iKey) || ( jKey == iKey && j < i );
      pos += (smaller)?inc:0;
      pos = min(pos,length);
    }
    int bits = 2*length-1; // mask for destination
    int dest = ((ii + pos) & bits) | (i & ~bits); // destination index in merged sequence
    //__syncthreads();
    key[dest] = iKey;
    value[dest] = iData;
    //__syncthreads();
  }
}


//! merges the two shared memory blocks so that key_dest will contain the minimum elements 
//! and key_src will contain the maximum elements
template< typename Key, typename Value >
__device__ void 
merge_block( Key* key_dest, Value *value_dest, Key* key_src, Value* value_src )
{
  int i = threadIdx.x;

  
  __shared__ bool changed;
  changed=false;
  __syncthreads();
  //sort_warp<float,int,true>(key_dest, value_dest);  
  if( key_dest[i] > key_src[i] )
  {
      float tmp = key_dest[i];
      key_dest[i]=key_src[i];
      key_src[i]=tmp;
      int tmpu = value_dest[i];
      value_dest[i]=value_src[i];
      value_src[i]=tmpu;
      changed=true;
  }
  __syncthreads();
  if( changed )
  {
      sort_warp<float,int,false>(key_src, value_src);
      sort_warp<float,int,true>(key_dest, value_dest);  
  }
  //sort_warp<float,int,false>(key_src, value_src);
}



}
}
#endif