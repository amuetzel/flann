/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011       Andreas Muetzel (amuetzel@uni-koblenz.de). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_KDTREE_CUDA_3D_INDEX_KERNELS_H_
#define FLANN_KDTREE_CUDA_3D_INDEX_KERNELS_H_

namespace flann
{

namespace KdTreeCudaPrivate
{
template< typename GPUResultSet, typename Distance, bool useWarpMerge >
struct searchNeighbors
{
    __device__
    void operator() ( const cuda::kd_tree_builder_detail::SplitInfo* splits,
                      const int* child1,
                      const int* parent,
                      const float4* aabbLow,
                      const float4* aabbHigh, const float4* elements, const float4& q, GPUResultSet& result, const Distance& distance = Distance() )
    {

        bool backtrack=false;
        int lastNode=-1;
        int current=0;
        if (!splits) // might be 0 to signal to the parallel kernel that a particular thread needs to be alive but not active
            return;

        cuda::kd_tree_builder_detail::SplitInfo split;
        while ( true )
        {
            if ( current==-1 ) break;
            split = splits[current];

            float diff1;
            if ( split.split_dim==0 ) diff1=q.x- split.split_val;
            else if ( split.split_dim==1 ) diff1=q.y- split.split_val;
            else if ( split.split_dim==2 ) diff1=q.z- split.split_val;

            // children are next to each other: leftChild+1 == rightChild
            int leftChild= child1[current];
            int bestChild=leftChild;
            int otherChild=leftChild;

            if ( ( diff1 ) <0 )
            {
                otherChild++;
            }
            else
            {
                bestChild++;
            }

            if ( !backtrack )
            {
                /* If this is a leaf node, then do check and return. */
                if ( leftChild==-1 )
                {
                    for ( int i=split.left; i<split.right; ++i )
                    {
                        float dist=distance.dist ( elements[i],q );
                        result.insert ( i,dist );
                    }
                    backtrack=true;
                    lastNode=current;
                    current=parent[current];
                }
                else   // go to closer child node
                {
                    lastNode=current;
                    current=bestChild;
                }
            }
            else   // continue moving back up the tree or visit far node?
            {
                // minimum possible distance between query point and a point inside the AABB
                float mindistsq=0;
                float4 aabbMin=aabbLow[otherChild];
                float4 aabbMax=aabbHigh[otherChild];

                if ( q.x < aabbMin.x ) mindistsq+=distance.axisDist ( q.x, aabbMin.x );
                else if ( q.x > aabbMax.x ) mindistsq+=distance.axisDist ( q.x, aabbMax.x );
                if ( q.y < aabbMin.y ) mindistsq+=distance.axisDist ( q.y, aabbMin.y );
                else if ( q.y > aabbMax.y ) mindistsq+=distance.axisDist ( q.y, aabbMax.y );
                if ( q.z < aabbMin.z ) mindistsq+=distance.axisDist ( q.z, aabbMin.z );
                else if ( q.z > aabbMax.z ) mindistsq+=distance.axisDist ( q.z, aabbMax.z );

                //  the far node was NOT the last node (== not visited yet) AND there could be a closer point in it
                if ( ( lastNode==bestChild ) && ( mindistsq <= result.worstDist() ) )
                {
                    lastNode=current;
                    current=otherChild;
                    backtrack=false;
                }
                else
                {
                    lastNode=current;
                    current=parent[current];
                }
            }

        }
    }
};

template< typename GPUResultSet, typename Distance >
struct searchNeighbors <GPUResultSet, Distance, true >
{
    __device__
    void operator() ( const cuda::kd_tree_builder_detail::SplitInfo* splits,
                      const int* child1,
                      const int* parent,
                      const float4* aabbLow,
                      const float4* aabbHigh, const float4* elements, const float4& q, GPUResultSet& result, const Distance& distance = Distance() )
    {

        bool backtrack=false;
        int lastNode=-1;
        int current=0;
        bool active=(splits!=0); // splits == 0 <-> this thread is part of the last thread block and doesn't have a query point

        cuda::kd_tree_builder_detail::SplitInfo split;
        while ( __any ( active ) )
        {
            //active = ( current!=-1 );
            int leftChild=0, bestChild, otherChild;
            if ( active )
            {
                // children are next to each other: leftChild+1 == rightChild
                leftChild= child1[current];
                split = splits[current];
                if( leftChild != -1 )
                {
                    bestChild=leftChild;
                    otherChild=leftChild;
                    

                    float diff1;
                    if ( split.split_dim==0 ) diff1=q.x- split.split_val;
                    else if ( split.split_dim==1 ) diff1=q.y- split.split_val;
                    else if ( split.split_dim==2 ) diff1=q.z- split.split_val;

                    if ( ( diff1 ) <0 )
                    {
                        otherChild++;
                    }
                    else
                    {
                        bestChild++;
                    }
                }
            }

            __shared__ int shared_index[32];
            __shared__ float shared_dist[32];
            __shared__ int leaf[32];
            __shared__ cuda::kd_tree_builder_detail::SplitInfo shared_split;
            __shared__ float4 shared_query;
            __shared__ GPUResultSet shared_result;

            // this part is a bit tricky to understand:
            // if any thread of this warp arrives at a leaf node, the "mode of parallelism" changes:
            // In sequence, every leaf node is searched by having all thread search the 
            // points in the node and then reducing the result and inserting it into the result set.
            // This is done for all threads that encountered a leaf node in this iteration.
            // Due to less divergence between the threads and a better memory access pattern, this leads to
            // a HUGE speedup!
            if ( __any ( leftChild==-1 ) )
            {
                // this can likely be replaced by __ballot, but i don't have the card for that
                leaf[threadIdx.x]= ( leftChild==-1 );

                for ( int l=0; l<32; l++ )
                {
                    if ( !leaf[l] )
                        continue;
                    //result.insert ( 0,0 );
                    shared_dist[threadIdx.x]=infinity();//result.worstDist();
                    shared_index[threadIdx.x]=-1;//result.bestIndex;
                    if ( l==threadIdx.x )
                    {
                        shared_split.left=split.left;
                        shared_split.right=split.right;
                        shared_query=q;
                        shared_result=result;
                    }
                    __syncthreads();

                    for ( int chunk_start=shared_split.left; chunk_start<shared_split.right; chunk_start+=32 )
                    {
                        int my_pos = chunk_start+threadIdx.x;
                        if ( my_pos< shared_split.right )
                        {
                            float4 query=shared_query;
                            float dist=distance.dist ( elements[my_pos],query );
                            if ( dist < shared_dist[threadIdx.x] )
                            {
                                shared_dist[threadIdx.x]=dist;
                                shared_index[threadIdx.x]=my_pos;
                            }
                        }
                        if (cuda::ResultSetTraits<GPUResultSet>::should_merge_after_each_block)
                        {
                            //shared_dist[threadIdx.x]=2;//result.worstDist();
                            //shared_index[threadIdx.x]=-1;//result.bestIndex;
                            result.mergeWarpElements(shared_index,shared_dist,0,&shared_result);
                            __syncthreads();
                            shared_dist[threadIdx.x]=infinity();//result.worstDist();
                            shared_index[threadIdx.x]=-1;//result.bestIndex;
                            if (threadIdx.x==l)
                                result=shared_result;
                            __syncthreads();
                        }
                    }
                    if (!cuda::ResultSetTraits<GPUResultSet>::should_merge_after_each_block)
                        result.mergeWarpElements(shared_index,shared_dist, l==threadIdx.x,&result);
                    
                }
            }

            if ( active )
            {
                if ( !backtrack )
                {

                    if ( leftChild==-1 )   // go to closer child node
                    {
                        backtrack=true;
                        lastNode=current;
                        current=parent[current];
                        if( current == -1 )
                            active=false;
                    }
                    else
                    {
                        lastNode=current;
                        current=bestChild;
                        
                    }
                }
                else   // continue moving back up the tree or visit far node?
                {
                    // minimum possible distance between query point and a point inside the AABB
                    float mindistsq=0;
                    float4 aabbMin=aabbLow[otherChild];
                    float4 aabbMax=aabbHigh[otherChild];

                    if ( q.x < aabbMin.x ) mindistsq+=distance.axisDist ( q.x, aabbMin.x );
                    else if ( q.x > aabbMax.x ) mindistsq+=distance.axisDist ( q.x, aabbMax.x );
                    if ( q.y < aabbMin.y ) mindistsq+=distance.axisDist ( q.y, aabbMin.y );
                    else if ( q.y > aabbMax.y ) mindistsq+=distance.axisDist ( q.y, aabbMax.y );
                    if ( q.z < aabbMin.z ) mindistsq+=distance.axisDist ( q.z, aabbMin.z );
                    else if ( q.z > aabbMax.z ) mindistsq+=distance.axisDist ( q.z, aabbMax.z );

                    //  the far node was NOT the last node (== not visited yet) AND there could be a closer point in it
                    if ( ( lastNode==bestChild ) && ( mindistsq <= result.worstDist() ) )
                    {
                        lastNode=current;
                        current=otherChild;
                        backtrack=false;
                    }
                    else
                    {
                        lastNode=current;
                        current=parent[current];
                        if( current == -1 )
                            active=false;
                    }
                }
            }

        }
    }
};


template< typename GPUResultSet, typename Distance >
__global__
void nearestKernel ( const cuda::kd_tree_builder_detail::SplitInfo* splits,
                     const int* child1,
                     const int* parent,
                     const float4* aabbMin,
                     const float4* aabbMax, const float4* elements, const float* query, int stride, int resultStride, int* resultIndex, float* resultDist, int querysize, GPUResultSet result, Distance dist = Distance() )
{
    typedef float DistanceType;
    typedef float ElementType;
    //                  typedef DistanceType float;
    size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
    size_t query_idx = tid;

    if ( tid >= querysize )
        query_idx = querysize-1;

    float4 q = make_float4 ( query[query_idx*stride],query[query_idx*stride+1],query[query_idx*stride+2],0 );

    result.setResultLocation ( resultDist, resultIndex, query_idx, resultStride );

    searchNeighbors<GPUResultSet, Distance, flann::cuda::ResultSetTraits<GPUResultSet>::has_warp_merge>() ( tid < querysize?splits:0,child1,parent,aabbMin,aabbMax,elements, q, result, dist );

    if( tid < querysize )
        result.finish();
}
}


//! thrust transform functor
//! transforms indices in the internal data set back to the original indices
struct map_indices
{
    const int* v_;

    map_indices ( const int* v ) : v_ ( v )
    {
    }

    __host__ __device__
    float operator() ( const int&i ) const
    {
        if ( i>= 0 ) return v_[i];
        else return i;
    }
};

//! implementation of L2 distance for the CUDA kernels
struct CudaL2
{

    static float
    __host__ __device__
    axisDist ( float a, float b )
    {
        return ( a-b ) * ( a-b );
    }

    static float
    __host__ __device__
    dist ( float4 a, float4 b )
    {
        float4 diff = a-b;
        return dot ( diff,diff );
    }
};

//! implementation of L1 distance for the CUDA kernels
//! NOT TESTED!
struct CudaL1
{

    static float
    __host__ __device__
    axisDist ( float a, float b )
    {
        return fabs ( a-b );
    }

    static float
    __host__ __device__
    dist ( float4 a, float4 b )
    {
        return fabs ( a.x-b.x ) +fabs ( a.y-b.y ) + ( a.z-b.z ) + ( a.w-b.w );
    }
};
}
#endif 
