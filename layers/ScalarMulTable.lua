--[[

 Input: A table {x, y} of a Tensor x and a scalar y.

 Output: x * y

--]]

local ScalarMulTable, parent = torch.class('nn.ScalarMulTable', 'nn.Module')

function ScalarMulTable:__init()
  parent.__init(self)
  self.gradInput = {}
end

function ScalarMulTable:updateOutput(input)
  local v, scale = unpack(input)

  self.output:resizeAs(v)
  -- bmm
  self.output = torch.bmm(
    scale:view(-1,1,1),
    v:view(-1,1,v:size(v:dim()))
  ):squeeze()
  --[[for i=1,scale:size(1) do
    self.output[i] = scale[i] * v[i]
  end]]

  return self.output
end

function ScalarMulTable:updateGradInput(input, gradOutput)
  local v, scale = unpack(input)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[2]:resizeAs(input[2])

  -- recover vector
  self.gradInput[1]:set(torch.bmm(
      gradOutput:view(-1,v:size(v:dim()),1),
      scale:view(-1,1,1)
    ):squeeze()
  )

  -- recover scale
  self.gradInput[2] = torch.bmm(
      gradOutput:view(-1,1,v:size(v:dim())),
      v:view(-1,v:size(v:dim()),1)
  ):squeeze():view(-1,1)

  --[[self.gradInput[1]:set(gradOutput * scale[1])
  self.gradInput[2][1] = gradOutput:dot(v)]]
  return self.gradInput
end
