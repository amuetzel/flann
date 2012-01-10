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
}
}
#endif