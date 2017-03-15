local MinNMask, parent = torch.class('nn.MinNMask', 'nn.Module')

-- let's use Min.lua's interface with a new "n" as input
function MinNMask:__init(n, dimension, nInputDims)
  parent.__init(self)
  n = n or 1
  self.n = n
  dimension = dimension or 1
  self.dimension = dimension
  self.nInputDims = nInputDims or 1
end

function MinNMask:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function MinNMask:_lazyInit()
   self._output = self._output or self.output.new()
   self._indices = self._indices or
      (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaLongTensor() or torch.LongTensor())
end


function MinNMask:updateOutput(input)
  self:_lazyInit()
  -- add one because we will assume we've got a batch dimension
  local dimension = self:_getPositiveDimension(input)
  local v_input
  if input:dim() == 1 then
    dimension = 2
    v_input = input:view(1,input:size(1))
  else
    v_input = input
  end


  torch.sort(self._output,self._indices, v_input, dimension)

  local zero_ind = self._indices:sub(1,v_input:size(1),1,self.n)
  self.output:resizeAs(v_input):zero()
  for j=1,v_input:size(1) do
    for i=1,self.n do
      self.output[j][zero_ind[j][i]] = 1
    end
  end

  self.output:squeeze(self.output)
  return self.output
end

function MinNMask:updateGradInput(input, gradOutput)
  self:_lazyInit()

  self.gradInput:resizeAs(gradOutput):zero()
  return self.gradInput
end

function MinNMask:type(type, tensorCache)
    self._indices = nil
    parent.type(self, type, tensorCache)
    return self
end

function MinNMask:clearState()
   nn.utils.clear(self, '_indices', '_output')
   return parent.clearState(self)
end
