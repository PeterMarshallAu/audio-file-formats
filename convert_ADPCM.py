#!/usr/bin/env python

"""
***
*** Convert ADPCM wave file to LINEAR PCM wave file
*** Only handles 1-channel (mono) and only converts to 16 bit.
***
*** The abovementioned format is useful for Google Speech API calls.
***
*** Pure Python implementation of format conversion so that it can
*** run under AppEngine.
***
*** Version: 0.9 (first GitHub version to publish)
*** Written for AppEngine's Python 2.7 environment
***
*** Author: Peter Marshall
***
***
*** For background on the algorithm, see:
*** http://www.drdobbs.com/database/algorithm-alley/database/sourcecode/algorithm-alley/30201952
*** https://bitbucket.org/python_mirrors/cpython/src/73024d00208f/Modules/audioop.c?at=default&fileviewer=file-view-default
*** http://soundfile.sapp.org/doc/WaveFormat/
***
*** TO DO
*** (nothing)
***
"""

import struct
from StringIO import StringIO


# Intel ADPCM step variation table.
index_table = [
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8 ]

step_size_table = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767 ]



# Ensure that a value is within a given range.
def clamp(value, lower_limit, upper_limit):
    return (lower_limit if value < lower_limit
       else upper_limit if value > upper_limit
       else value)


# Decompresses a single packet from MS ADPCM to Linear PCM 16 bit.
def convert_packet(data_in, valpred, index):
    size = 2   # always 2 bytes (16 bits) per sample
    data_in_len = len(data_in)
    data_in_index = 0

    # 2 bytes per output sample * one byte of input yields two samples * length
    # of input buffer in bytes.
    # Plus one extra sample that we get from the state passed in.
    data_out_len = size * 2 * data_in_len + 2
    data_out = bytearray(data_out_len)

    # First sample is what was passed in as state.
    # That value came from the packet header.
    data_out[0:2] = struct.pack('<h', valpred)

    buffer_step = True
    for i in range(2, data_out_len, size):

        # Get the delta value.
        # Buffer step is an alternating boolean that indicates either fetching a new
        # byte or reading from existing byte (because each sample is a 4 bit value).
        # Low order nybble is processed first, then high order nybble.
        if not buffer_step:
            delta = (input_buffer >> 4) & 0x0f
        else:
            input_buffer = ord(data_in[data_in_index])
            data_in_index += 1
            delta = input_buffer & 0x0f
        buffer_step = not buffer_step

        # Set current step size.
        step_size = step_size_table[index]

        # Find new index value (for next iteration).
        index += index_table[delta]
        index = clamp(index, 0, 88)

        # Separate sign and magnitude.
        # The high bit is the sign and the rest is the magnitude.
        sign = delta & 8
        delta = delta & 7

        # Compute difference and new predicted value.
        difference = (step_size >> 3);
        if delta & 1:
            difference += (step_size >> 2)
        if delta & 2:
            difference += (step_size >> 1)
        if delta & 4:
            difference += step_size
        if sign:
            valpred -= difference
        else:
            valpred += difference

        # Clamp output value.
        valpred = clamp(valpred, -32768, 32767)

        # Add the output sample to buffer.
        data_out[i:i+2] = struct.pack('<h', valpred)

    return data_out, valpred, index


# Read the data chunk of a WAV file, decompressing each packet
# and writing output.
# The input file must already by positioned at the beginning
# of the data chunk.
def convert_file(file_in, file_out, data_in_len, block_align):
    assert data_in_len > 0 and block_align > 0
    bytes_read = 0
    bytes_written = 0

    while bytes_read < data_in_len:
        this_packet = file_in.read(block_align)
        bytes_read += len(this_packet)

        # First two bytes of packet is the first sample value.
        # Third byte is the index in step table of the current step.
        # Fourth byte is empty.
        # Remaining bytes of packet are the subsequent sample values.
        first_sample = struct.unpack('<h', this_packet[0:2])[0]
        index = ord(this_packet[2])
        packet_data = this_packet[4:]

        packet_out, _, _ = convert_packet(packet_data, first_sample, index)

        file_out.write(packet_out)
        bytes_written += len(packet_out)

    return bytes_written


# Given a WAV file, position file to start of data chunk and determine length.
# Write all necessary headers for the output file.
def wav_find_data_chunk(file_in, file_out):
    # Read WAVE and fmt headers.
    front_headers = file_in.read(36)

    # A valid WAV file will have standard headers at beginning. 
    if not (front_headers[0:4] == 'RIFF' and
            front_headers[8:12] == 'WAVE' and 
            front_headers[12:16] == 'fmt '): 
         raise Exception('Not a valid WAV file.')

    audio_format = struct.unpack('<H', front_headers[20:22])[0]
    num_channels = struct.unpack('<H', front_headers[22:24])[0]
    sample_rate = struct.unpack('<L', front_headers[24:28])[0]
    block_align = struct.unpack('<H', front_headers[32:34])[0]

    # We can only process a 1-channel (mono) MS ADPCM file. 
    if not (num_channels == 1 and audio_format in [2,17]): 
         raise Exception('Only a 1-channel (mono) MS ADPCM file can be processed.')

    # Amend the fmt header for linear PCM.
    # Write new front headers as output.
    new_headers = bytearray(front_headers)
    new_headers[20:22] = [1, 0]    # linear PCM
    new_headers[32:34] = [2, 0]    # block alignment in bytes
    new_headers[34:36] = [16, 0]   # bits per sample
    new_headers[16:20] = struct.pack('<L', 16)    # fmt chunk size
    new_headers[28:32] = struct.pack('<L', 2 * sample_rate)    # byte rate
    file_out.write(new_headers)

    # Determine total size of input file.
    file_in.seek(0, 2)   # to EOF
    in_file_size = file_in.tell()

    # Iterate through chunks until we find the data chunk.
    next_chunk_pos = 12   # fixed location of first chunk
    while True:
        file_in.seek(next_chunk_pos)
        chunk_header = file_in.read(8)
        chunk_type = chunk_header[0:4]
        chunk_length = struct.unpack('<L', chunk_header[4:8])[0]

        if chunk_type == 'data':
            file_out.write(chunk_header)
            return chunk_length, block_align

        next_chunk_pos += 8 + chunk_length
        if next_chunk_pos >= in_file_size:
            raise Exception('Not a valid WAV file.')


# Main flow of control.
def convert_linear16(file_name_in, file_name_out, file_in_data = None):
    # Either use real files or memory files using StringIO.
    # Memory file is useful for AppEngine, which has a read-only file system.
    # !! Beware of file names chosen - there is no protection against
    # !! overwriting a file.
    if file_name_in:
        file_in = open(file_name_in,'rb')
    else:
        file_in = StringIO(file_in_data)
    if file_name_out:
        file_out = open(file_name_out,'wb')
    else:
        file_out = StringIO()

    data_in_len, block_align = wav_find_data_chunk(file_in, file_out)
    bytes_written = convert_file(file_in, file_out, data_in_len, block_align)

    # Write the data chunk length.
    file_out.seek(40)
    file_out.write(struct.pack('<L', bytes_written))

    # Write the entire WAV file's chunk length.
    file_out.seek(0, 2)    # to EOF
    out_file_size = file_out.tell()
    file_out.seek(4)
    file_out.write(struct.pack('<L', out_file_size - 8))

    file_in.close()
    if file_name_out:
        file_out.close()
    else:
        # StringIO result can be referenced in calling function.
        return file_out.getvalue()


# Entry point.
if __name__ == '__main__':
    convert_linear16('in_file.wav', 'out_file.wav')
