def flops(image_size, batch): 
    def conv_flops(dim_in, dim_out, S, kernel_size, stride):
        return dim_in* dim_out* (S  / stride ) **2 * (2* kernel_size**2 +1)
    def maxpooling_flops(stride, kernel_size, dim_out, S):
        return kernel_size**2 * (S /2)**2 * dim_out
    def gap_flops(S):
        return 512* S**2
    def linear_flops(l_in, l_out):
        return 2* l_in*l_out

    total_flops = 0
    total_flops += conv_flops(3, 32, image_size/2, 7,2)
    total_flops += maxpooling_flops(2,7,32,image_size/4)
    total_flops += conv_flops(32, 64, image_size/4, 5,1)
    total_flops += conv_flops(64, 128, image_size/8, 3,2)
    total_flops += conv_flops(128, 256, image_size/8,1,1) 
    total_flops += conv_flops(256, 256, image_size/16, 3,2)
    total_flops += conv_flops(256, 512, image_size/16,1,1)   
    
    total_flops+=gap_flops(image_size/16)
    total_flops+=linear_flops(512,256)
    total_flops+=linear_flops(256,100)
    return batch* total_flops# -> float (FLOPs)
def memory(image_size, B): 
    fp32_size = 32

    def mem_conv(channels_in, channels_out, kernel_size):
        return kernel_size**2 * channels_in * channels_out

    def mem_maxpool(c, S):
        return c* S**2

    def mem_gap( S):
        return  S**2
    def mem_tensor(c, S):
        return c* S**2

    def mem_linear(ch_in, ch_out):
        return ch_in* ch_out

        
    max_mem =0
    max_mem += mem_tensor(3,S)
    
    max_mem += mem_conv(3, 32,  7)

    max_mem += mem_tensor(3,S/2)
    max_mem += mem_maxpool(32,S/2)

    max_mem += mem_tensor(3,S/4)
    max_mem += mem_conv(32, 64,   5)

    max_mem += mem_tensor(3,S/8)
    max_mem += mem_conv(64, 128,  3)

    max_mem += mem_tensor(3,S/8)
    max_mem += mem_conv(128, 256, 1)

    max_mem += mem_tensor(3,S/16)
    max_mem += mem_conv(256, 256, 3)

    max_mem += mem_tensor(3,S/16)
    max_mem += mem_conv(256, 512, 1)  
    
    max_mem += mem_gap(512)
    max_mem +=  mem_linear(512,256)
    max_mem +=  mem_linear(256,100)

    return max_mem * B / 8 # -> float (bytes)

def bytes_moved(S,B):

    float32_size = 32
    def bytes_moved_conv(S, stride, channels_in, channels_out, kernel_size):
        return kernel_size**2 * channels_in + S **2 * channels_in +  (S/stride**2) * channels_out

    def bytes_moved_maxpool(c, S):
        return S **2 * c + (S/2)**2 * c

    def bytes_moved_gap( S, c):
        return  S**2 * c + c
        
    def bytes_moved_relu(c, S):
        return c *S**2 +c*S**2 

    def bytes_moved_linear(ch_in, ch_out):
        return ch_in * ch_out + ch_in + ch_out

    total_bytes_moved = 0
    
    total_bytes_moved += bytes_moved_conv(S/2, 2, 3, 32, 7)
    total_bytes_moved += bytes_moved_maxpool(32, S/2)
    total_bytes_moved += bytes_moved_relu(32, S/4)
    
    total_bytes_moved += bytes_moved_conv(S/4, 1, 32, 64, 5)
    total_bytes_moved += bytes_moved_relu(64, S/4)
    
    total_bytes_moved += bytes_moved_conv(S/8, 2, 64, 128, 3)
    total_bytes_moved += bytes_moved_relu(128, S/8)
    
    total_bytes_moved += bytes_moved_conv(S/8, 1, 128, 256, 1)
    total_bytes_moved += bytes_moved_relu(256, S/8)
    
    total_bytes_moved += bytes_moved_conv(S/16, 2, 256, 256, 3)
    total_bytes_moved += bytes_moved_relu(256, S/16)
    
    total_bytes_moved += bytes_moved_conv(S/16, 1, 256, 512, 1)
    total_bytes_moved += bytes_moved_relu(512, S/16)
    
    total_bytes_moved += bytes_moved_gap(S/16, 512)
    total_bytes_moved += bytes_moved_linear(512, 256)
    total_bytes_moved += bytes_moved_linear(256, 100)

    return total_bytes_moved * B * float32_size / 8
    
def latency(image_size, batch, theta=(1,1)):
    peakFlops = 8.1e12                    # FLOP/s
    bandwidth = 32 * 10**9 / 8            # 4e9 bytes/s = 32 Gbit/s
    return max(
        theta[0] * flops(image_size, batch) / peakFlops,
        theta[1] * bytes_moved(image_size, batch) / bandwidth,
    )
        # -> float (seconds)
def energy(image_size, batch, theta_energy=(1.0, 1.0)):
    e_per_flop = 4e-12     # Дж/FLOP
    e_per_byte = 20e-12    # Дж/байт
    E_flops = flops(image_size, batch) * e_per_flop
    E_bytes = bytes_moved(image_size, batch) * e_per_byte
    return max(theta_energy[0] * E_flops, theta_energy[1] * E_bytes)   # -> float (joules)