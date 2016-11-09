require 'torch'
require "optim"
require "nn"
require 'xlua'
require 'cutorch'
require "cunn"
require "cudnn"
require 'paths'

m2=require 'minc2_simple'

torch.setdefaulttensortype('torch.FloatTensor')


-- setup input data
hc_prefix='./'
hc_list=hc_prefix..'small_10.lst'
-- hc_list=hc_prefix..'small_all.lst'
hc_samples={}


-- mlp parameters
HUs=100       -- number of neurons
fov=8        -- fov in pixels, patches are (fov*2)**3
iter=100       -- number of optimization iterations, for each minibatch 
l1=5         -- layer 1 kernel size
l2=3         -- layer 2 kernel size
maps1=32
maps2=32
LR=0.001      -- learning rate
momentum=0.9 -- momentum
WD=5e-4      -- weight decay
train=8      -- use first N subjects for training 
mult=1       -- how many datasets to include in a single training
test=10      -- subject for testing
batches=10   -- number of training batches
stride=4
-- seed RNG
torch.manualSeed(0)


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

dataset={}

for _,l in pairs(hc_samples) do
    print(string.format("Opening %s %s",l[1],l[2]))
    
    local t1=m2.minc2_file.new(l[1])
    t1:setup_standard_order()
    
    local seg=m2.minc2_file.new(l[2])
    seg:setup_standard_order()
    
    dataset[#dataset+1]={ t1:load_complete_volume(m2.MINC2_FLOAT), 
                          seg:load_complete_volume(m2.MINC2_INT) }
end

t1_mean=0.0
t1_sd=0.0

print("removing mean and sd")

for j=1,(#dataset) do
    t1_mean=t1_mean+torch.mean(dataset[j][1])
    t1_sd=t1_sd+torch.std(dataset[j][1])
end

t1_mean=t1_mean/#dataset
t1_sd=t1_sd/#dataset
print(string.format("Mean=%f sd=%f",t1_mean,t1_sd))

for j=1,(#dataset) do
    dataset[j][1]=(dataset[j][1]-t1_mean)/t1_sd
end


-- convert volumes into overlapping tiles and create a 4D minibatch
-- TODO: use stride
local function get_tiles(minibatch, dataset, train,  fov, stride, mult, use_rnd)
    
    volume_sz=dataset[1][1]:size()
    patch=fov*2
    
    out_el = ( math.floor((volume_sz[1]-patch+stride-1)/stride) )*
             ( math.floor((volume_sz[2]-patch+stride-1)/stride) )*
             ( math.floor((volume_sz[3]-patch+stride-1)/stride) )*mult
    
    -- minibatch
    -- out1=torch.Tensor(out_el,patch,patch,patch)
    -- out2=torch.LongTensor(out_el)
    
    idx=1
    pidx={{1,1},{1,1},{1,1}}
    
    rrr=torch.IntTensor(out_el)
    dx=torch.IntTensor(out_el,3)
    
    if use_rnd then
        rrr:random(1,train)
        dx:random(0,stride-1)
    else
        rrr:fill(train)
        dx:fill(0)
    end
    
    -- TODO: avoid creating temporary tensors somehow?
    out_image=torch.FloatTensor(out_el,1,patch,patch,patch)
    out_label=torch.ByteTensor(out_el)
    for m=1,mult do
        for i=(1+fov),(volume_sz[1]-fov),stride do
            for j=(1+fov),(volume_sz[2]-fov),stride do
                for k=(1+fov),(volume_sz[3]-fov),stride do
                    
                    pidx[1]={i-fov,i+fov-1}
                    pidx[2]={j-fov,j+fov-1}
                    pidx[3]={k-fov,k+fov-1}
                    
                    -- add randomness
                    if k<(volume_sz[3]-stride-fov) and 
                       j<(volume_sz[2]-stride-fov) and 
                       i<(volume_sz[1]-stride-fov) then
                        
                        pidx[1][1]=pidx[1][1]+dx[{idx,1}]
                        pidx[1][2]=pidx[1][2]+dx[{idx,1}]
                        
                        pidx[2][1]=pidx[2][1]+dx[{idx,2}]
                        pidx[2][2]=pidx[2][2]+dx[{idx,2}]
                        
                        pidx[3][1]=pidx[3][1]+dx[{idx,3}]
                        pidx[3][2]=pidx[3][2]+dx[{idx,3}]
                        
                    end
                    
                    out_image[idx][1]=dataset[ rrr[idx] ][1][pidx]
                    out_label[idx]   =dataset[ rrr[idx] ][2][{i,j,k}]+1 -- convert to 1-based class id 
                    
                    idx=idx+1
                end
            end
        end
    end
    -- print(idx,out_el)
--     print(minibatch[1]:size())
--     print(out_image:size())
--     print("---")
--     
--     minibatch[1]:copy(out_image)
--     minibatch[2]:copy(out_label)
--     
--     print(minibatch[1]:size())
--     print(minibatch[2]:size())
--     print("===")
end

local function allocate_tiles(ds, fov, stride,mult)
    volume_sz=ds[1]:size()
    patch=fov*2
    
    out_el = ( math.floor((volume_sz[1]-patch+stride-1)/stride) )*
             ( math.floor((volume_sz[2]-patch+stride-1)/stride) )*
             ( math.floor((volume_sz[3]-patch+stride-1)/stride) )*mult
    
    -- minibatch
    out1=torch.CudaTensor(out_el,1,patch,patch,patch)
    out2=torch.CudaByteTensor(out_el)
    
    print(string.format("Training dataset:%d elements",out_el*patch*patch*patch))

    return {out1,out2}
end

local function put_tiles(ds, out, fov, stride, mult, ch)
    
    local volume_sz=ds[1]:size()
    
    local patch=fov*2
    local out_el=(volume_sz[1]-patch)*(volume_sz[2]-patch)*(volume_sz[3]-patch)
    -- TODO: verify size of out and out_el
    
    local out_t=torch.Tensor(volume_sz):fill(0.0)
    
    out_t[{{1+fov,volume_sz[1]-fov},{1+fov,volume_sz[2]-fov},{1+fov,volume_sz[3]-fov}}] = 
        out:float():view(mult,volume_sz[1]-patch, volume_sz[2]-patch, volume_sz[3]-patch, 2)[{1,{},{},{},ch}]:exp()
    
    return out_t
end

local function put_tiles_max(ds, out, fov, stride,mult)
    
    local volume_sz=ds[1]:size()
    
    local patch=fov*2
    local out_el=(volume_sz[1]-patch)*(volume_sz[2]-patch)*(volume_sz[3]-patch)
    -- TODO: verify size of out and out_el
    
    local out_t=torch.ByteTensor(volume_sz):fill(0)
    
    _,out_t[ {{1+fov,volume_sz[1]-fov},{1+fov,volume_sz[2]-fov},{1+fov,volume_sz[3]-fov}} ] = 
        out:float():view(mult,volume_sz[1]-patch, volume_sz[2]-patch, volume_sz[3]-patch,2)[{1,{},{},{},{}}]:max(4)
    
    return out_t
end


patch=fov*2 -- size of input chunks

patch_l1=patch - l1+1
patch_l2=(patch_l1)/2-l2+1


-- prepare network
model = nn.Sequential()  -- make a multi-layer perceptron with a single output 
model:add(nn.VolumetricConvolution(1,maps1,l1,l1,l1)) -- converts patch**3 -> patch_l1**3
model:add(nn.Tanh())
model:add(nn.VolumetricMaxPooling(2,2,2))             -- patch_l1 -> patch_l1/2
model:add(nn.VolumetricConvolution(maps1,maps2,l2,l2,l2)) -- patch_l1/2 -> patch_l2
model:add(nn.Tanh())
model:add(nn.Reshape( maps2*patch_l2*patch_l2*patch_l2 ))
model:add(nn.Linear(  maps2*patch_l2*patch_l2*patch_l2, HUs))

model:add(nn.Dropout(0.5))
model:add(nn.Tanh())
model:add(nn.Linear(HUs,2))
model:add(nn.LogSoftMax())
cudnn.convert(model, cudnn)

print(model)

model=model:cuda()

criterion = nn.ClassNLLCriterion()
criterion=criterion:cuda()


minibatch=allocate_tiles(dataset[1],fov,stride,mult) -- allocate data in GPU

print(string.format("Running optimization using %d batches and %d iterations per batch",batches,iter))

parameters, gradParameters = model:getParameters()
timer = torch.Timer()

model:training()
for j = 1,batches do
    -- reset optimization state here
    optimState = {
        learningRate = LR,
        learningRateDecay = 0.0,
        momentum = momentum,
        dampening = 0.0,
        weightDecay = WD
    }
    
    timer:reset()
    -- generate random samples from training dataset
    get_tiles(minibatch,dataset,train,fov,stride,mult,true)
    
    load_time=timer:time().real
    --print(string.format("Data loading:%f",timer:time().real))
    timer:reset()
    model_name=string.format('cnn_training_%03d.t7',j)
    
    if paths.filep(model_name) then
        model=torch.load(model_name)
        print("Loaded model:"..model_name)
        print(model)
    else
     
        local avg_err=0
        -- xlua.progress(0,iter)
        for i=1,iter do
            local err, outputs
            feval = function(x)
                model:zeroGradParameters()
                outputs = model:forward(minibatch[1])
                err = criterion:forward(outputs, minibatch[2])
                local gradOutputs = criterion:backward(outputs, minibatch[2])
                model:backward( minibatch[1], gradOutputs)
                return err, gradParameters
            end
            optim.sgd(feval, parameters, optimState)
            -- avg_err=avg_err+err
            -- if i%20 ==0 then xlua.progress(i,iter) end
            print(err)
        end
        model:clearState()
        -- torch.save(model_name,model)
        print(string.format("%d proc %f sec, load: %f sec, avg err:%f",j,timer:time().real,load_time,avg_err/iter))
    end
end

model:evaluate()
get_tiles(minibatch,dataset,test,fov,1,mult,false)
out1=model:forward(minibatch[1])
err1=criterion:forward(out1, minibatch[2])
print(string.format("Error on test dataset:%e",err1))
print(out1:size())

t_out1=put_tiles(dataset[1],out1,fov,1,mult,1)
-- t_out2=put_tiles(dataset[1],out1,fov,1,2)
t_out2=put_tiles_max(dataset[1],out1,fov,1,mult)

-- reference
t1=m2.minc2_file.new(hc_samples[1][1])
out_minc=m2.minc2_file.new()

out_minc:define(t1:store_dims(), m2.MINC2_BYTE, m2.MINC2_FLOAT)
out_minc:create("output1.mnc")
out_minc:setup_standard_order()
out_minc:save_complete_volume(t_out1)

out_minc=m2.minc2_file.new()
out_minc:define(t1:store_dims(), m2.MINC2_BYTE, m2.MINC2_BYTE)
out_minc:create("output2.mnc")
out_minc:setup_standard_order()
out_minc:save_complete_volume(t_out2)

out_minc=m2.minc2_file.new()
out_minc:define(t1:store_dims(), m2.MINC2_BYTE, m2.MINC2_FLOAT)
out_minc:create("output3.mnc")
out_minc:setup_standard_order()
out_minc:save_complete_volume(dataset[1][1]:float())
