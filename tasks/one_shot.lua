require('../')
require('./util')
require('optim')
--require 'xlua'

-- load NTM model and criterion w/ config
local config = {
  gpu = true,
  output_size = 3,
  symbol_size = 8,
  learningRate = 1e-4,
  optim = 'adam',
}

local gpu = config.gpu or true
local learningRate = config.learningRate or 1e-3

local output_size = config.output_size or 3
local symbol_size = config.symbol_size or 3
local batch_size = config.batch_size or 16
local input_size = config.input_size or symbol_size+output_size

local ntm_config = {
  input_dim = input_size,
  output_dim = output_size,
  mem_rows = config.mem_rows or 10,
  mem_cols = config.mem_cols or 8,
  cont_dim = config.cont_dim or 50,
  use_lrua = config.use_lrua or true,
  use_cuda = gpu,
  read_heads = config.read_heads or 4,
  write_heads = config.write_heads or 4,
  batch_size = batch_size
}

local model = ntm.NTM(ntm_config)

if gpu then model:cuda() end

local criteria = nn.ClassNLLCriterion()
if gpu then criteria:cuda() end

params, grads = model:getParameters()

local episode,targets
if gpu then
  episode = torch.CudaTensor()
  targets = torch.CudaTensor()
else
  episode = torch.Tensor()
  targets = torch.Tensor()
end

local l_optim, optim_config
if config.optim == 'adam' then
  require 'optim'
  l_optim = optim.adam
  optim_config = {
    learningRate = config.learningRate or 1e-3
  }
else
  l_optim = ntm.rmsprop
  optim_config = {
    learningRate = config.learningRate or 1e-3,
    momentum = config.momentum or 0.9,
    decay = config.decay or 0.95
  }
end

local forward = function(mod, crit, epi, tar_hot)
  local loss = 0
  local num_batches = epi:size(2)
  local output_size = tar_hot:size(3)
  local _,tar = tar_hot:max(tar_hot:dim())
  tar = tar:view(-1)
  if mod.use_cuda then tar = tar:cuda() end
  local episode_length = epi:size(1)
  local out
  if mod.use_cuda then
    out = torch.CudaTensor(episode_length, num_batches, output_size)
  else
    out = torch.Tensor(episode_length, num_batches, output_size)
  end
  for i = 1,episode_length do
    out[i] = mod:forward(epi[i])
  end

  loss = crit:forward(out:view(-1,output_size), tar) --* input_size

  return out, loss
end

local backward = function(mod, crit, tar_hot, out, epi)
  local episode_length = out:size(1)
  local num_batches = epi:size(2)
  local output_size = tar_hot:size(3)
  local _,tar = tar_hot:max(tar_hot:dim())
  tar = tar:view(-1)

  if mod.use_cuda then tar = tar:cuda() end

  local gradOutputs = crit:backward(out:view(-1,output_size),tar)

  gradOutputs = gradOutputs:view(episode_length, num_batches, output_size)

  -- don't need to have an error for last cell (can't know target)
  for i = episode_length, 2, -1 do
    mod:backward(
      epi[i],
      gradOutputs[i]
        --:mul(input_size)
    )

  end
end

-- load a batch externally and then update model with it
local all_loss = {}
local trainBatch = function(episodeCPU,targetsCPU)
  local b_loss = 0
  model:forget()

  episode:resize(episodeCPU:size()):copy(episodeCPU)
  targets:resize(targetsCPU:size()):copy(targetsCPU)

  local feval = function(x)
    local loss = 0
    model:zeroGradParameters()
    -- forward each input for episode
    local outputs, sample_loss = forward(model, criteria, episode, targets)
    loss = loss + sample_loss
    b_loss = loss
    backward(model, criteria, targets, outputs, episode)
    -- clip gradients
    grads:clamp(-10, 10)
    --print('Loss: '..loss..' iter: '..iter..'epi: '..i)
    return loss, grads
  end
  l_optim(feval, params, optim_config)
  table.insert(all_loss,b_loss)
  return b_loss
end

-- generate simple data toy problem for debugging
-- symbol... labels -> from generator
-- ({0,1,0},{0,0,1})
local generateBatch = function(num_class, symbol_size, num_examples, batch_size)
  -- generate batch_size episodes and fold
  local num_episodes = num_class*num_examples
  local targets = torch.Tensor(num_episodes, batch_size, num_class):zero()
  local samples = torch.Tensor(num_episodes, batch_size, symbol_size + num_class):zero()

  -- simpler
  local symbol = torch.rand(batch_size, num_class, symbol_size)--*1e-1

  for i=1,num_episodes do
    for j=1,batch_size do
      local label_ind = torch.randperm(num_class)[1]
      targets[i][j][label_ind] = 1
      -- keep zero if first episode
      if i > 1 then
        samples[i][j][{{1+symbol_size,symbol_size+num_class}}] = targets[i-1][j]
      end
      samples[i][j][{{1,symbol_size}}] = torch.add(
        symbol[j][label_ind],
        (torch.rand(symbol_size)*1e-3):zero()
      )
    end
  end
  return samples,targets
end

local loss_m_avg = 0
-- iterator
local iters = 50000
for i=1,iters do
  local loss = 0

  local episodeCPU,targetsCPU = generateBatch(output_size,symbol_size,10,16)
  trainBatch(episodeCPU,targetsCPU)

  local alpha = 0.1
  local latest_loss = all_loss[#all_loss]
  if loss_m_avg == 0 then loss_m_avg = latest_loss end
  loss_m_avg = alpha*latest_loss + (1-alpha)*loss_m_avg
  print(loss_m_avg)

  --xlua.progress(i,iters)

end
