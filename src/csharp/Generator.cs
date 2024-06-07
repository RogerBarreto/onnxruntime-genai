// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Generator : IDisposable
    {
        private IntPtr _generatorHandle;
        private bool _disposed = false;

        public Generator(Model model, GeneratorParams generatorParams)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateGenerator(model.Handle, generatorParams.Handle, out _generatorHandle));
        }

        public bool IsDone()
        {
            return NativeMethods.OgaGenerator_IsDone(_generatorHandle);
        }

        public void ComputeLogits()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_ComputeLogits(_generatorHandle));
        }

        public void GenerateNextToken()
        {
            Result.VerifySuccess(NativeMethods.OgaGenerator_GenerateNextToken(_generatorHandle));
        }

#if NET8_0_OR_GREATER || NETSTANDARD2_1_OR_GREATER
        public ReadOnlySpan<int> GetSequence(ulong index)
#else
        public int[] GetSequence(ulong index)
#endif
        {
            ulong sequenceLength = NativeMethods.OgaGenerator_GetSequenceCount(_generatorHandle, (UIntPtr)index).ToUInt64();
            IntPtr sequencePtr = NativeMethods.OgaGenerator_GetSequenceData(_generatorHandle, (UIntPtr)index);
            unsafe
            {
#if NET8_0_OR_GREATER || NETSTANDARD2_1_OR_GREATER
                return new ReadOnlySpan<int>(sequencePtr.ToPointer(), (int)sequenceLength);
#else

                int[] sequence = new int[sequenceLength];
                for (int i = 0; i < (int)sequenceLength; i++)
                {
                    sequence[i] = Marshal.ReadInt32(sequencePtr, i * sizeof(int));
                }
                return sequence;
#endif
            }
        }

        ~Generator()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            NativeMethods.OgaDestroyGenerator(_generatorHandle);
            _generatorHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
