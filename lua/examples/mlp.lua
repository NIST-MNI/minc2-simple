require 'torch'
m2=require 'minc2_simple'
torch.setdefaulttensortype('torch.FloatTensor')
require "nn"


-- setup input data
hc_prefix='./'
hc_list=hc_prefix..'small_10.lst'
-- hc_list=hc_prefix..'small_all.lst'
hc_samples={}


-- load HC training list
for line in io.lines(hc_list) do
    sample={}
    for i,j in pairs(string.split(line,",")) do
        sample[#sample+1]=hc_prefix..j
    end
    hc_samples[#hc_samples + 1] = sample
end
  
print(#hc_samples)
-- load minc files into memory

t1s={}
segs={}
for _,l in pairs(hc_samples) do
    local t1=m2.minc2_file.new(l[1])
    t1:setup_standard_order()
    local seg=m2.minc2_file.new(l[2])
    seg:setup_standard_order()
    
    t1s[#t1s+1]=t1:load_complete_volume(m2.MINC2_FLOAT)
    segs[#t1s+1]=seg:load_complete_volume(m2.MINC2_BYTE)
end


-- will create a simple class for loading data into optimizer
    Dataset = torch.class('Dataset')
    
    function Dataset:__init(t1s,segs,field) 
        self.t1s=t1s
        self.segs=segs
        self.field=field
        
        for i,j in pairs(t1s) do
            self[i]={t1s[i],segs[i]}
        end
    end
    
    function Dataset:size()
        return #self.t1s
    end


dataset = Dataset.new(t1s,segs)

-- tensor sizes
sz=t1s[1]:size()
el=t1s[1]:storage():size()
HUs=10
-- prepare network
mlp = nn.Sequential()  -- make a multi-layer perceptron
mlp:add(nn.Reshape(el))
mlp:add(nn.Linear(el,HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs,el))
mlp:add(nn.Reshape(sz))

criterion = nn.MSECriterion()

-- do some funky processing

-- out=mlp:forward(t1s[1])

for i = 1,2500 do
    local input= dataset[i][1]
    local output = dataset[i][2]
    
    -- feed it to the neural network and the criterion
    criterion:forward(mlp:forward(input), output)
    
    
    mlp:zeroGradParameters()
    mlp:backward(input, criterion:backward(mlp.output, output))
    mlp:updateParameters(0.01)
    print(i)
end
    

local t1=m2.minc2_file.new(hc_samples[1][1])
local out_minc=m2.minc2_file.new()
out_minc:define(t1:store_dims(), m2.MINC2_BYTE, m2.MINC2_FLOAT)
out_minc:create("output.mnc")
out_minc:setup_standard_order()
out_minc:save_complete_volume(out)

