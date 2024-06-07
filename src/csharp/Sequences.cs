// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Sequences : IDisposable
    {
        private IntPtr _sequencesHandle;
        private bool _disposed = false;
        private ulong _numSequences;

        internal Sequences(IntPtr sequencesHandle)
        {
            _sequencesHandle = sequencesHandle;
            _numSequences = NativeMethods.OgaSequencesCount(_sequencesHandle).ToUInt64();
        }

        internal IntPtr Handle { get { return _sequencesHandle; } }

        public ulong NumSequences { get { return _numSequences; } }

#if NET8_0_OR_GREATER || NETSTANDARD2_1_OR_GREATER
        public ReadOnlySpan<int> this[ulong sequenceIndex]
#else
        public int[] this[ulong sequenceIndex]
#endif
        {
            get
            {
                if (sequenceIndex >= _numSequences)
                {
                    throw new ArgumentOutOfRangeException(nameof(sequenceIndex));
                }
                ulong sequenceLength = NativeMethods.OgaSequencesGetSequenceCount(_sequencesHandle, (UIntPtr)sequenceIndex).ToUInt64();
                IntPtr sequencePtr = NativeMethods.OgaSequencesGetSequenceData(_sequencesHandle, (UIntPtr)sequenceIndex);
                unsafe
                {
#if NET8_0_OR_GREATER || NETSTANDARD2_1_OR_GREATER
                    return new ReadOnlySpan<int>(sequencePtr.ToPointer(), (int)sequenceLength);
#else
                    int[] sequence = new int[sequenceLength];
                    Marshal.Copy(sequencePtr, sequence, 0, (int)sequenceLength);
                    return sequence;
#endif
                }
            }
        }

        ~Sequences()
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
            NativeMethods.OgaDestroySequences(_sequencesHandle);
            _sequencesHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
