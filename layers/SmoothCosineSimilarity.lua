--[[

Input: a table of two inputs {M, k}, where
  M = an n-by-m matrix
  k = an m-dimensional vector

Output: an n-dimensional vector

Each element is an approximation of the cosine similarity between k and the
corresponding row of M. It's an approximation since we add a constant to the
denominator of the cosine similarity function to remove the singularity when
one of the inputs is zero.

--]]

local SmoothCosineSimilarity, parent = torch.class('nn.SmoothCosineSimilarity', 'nn.Module')

function SmoothCosineSimilarity:__init(smoothen)
  parent.__init(self)
  self.gradInput = {}
  self.smooth = smoothen or 1e-3
end

function SmoothCosineSimilarity:updateOutput(input)
  local M, k = unpack(input)
  self.rownorms = torch.cmul(M, M):sum(M:dim()):sqrt():view(-1,M:size(M:dim()-1))
  self.knorm = torch.bmm(
    k:view(-1,1,k:size(k:dim())),
    k:view(-1,k:size(k:dim()),1)
  ):sqrt():squeeze()
  --self.dot = M * k --bmm
  self.dot = torch.bmm(M,k:view(-1,k:size(k:dim()),1)):squeeze()
  local denom = torch.cmul(
    self.knorm:view(-1,1):expand(M:size(1),
    M:size(2)),self.rownorms
  )
  self.output:set(torch.cdiv(self.dot, denom:add(self.smooth)))
  return self.output
end

function SmoothCosineSimilarity:updateGradInput(input, gradOutput)
  local M, k = unpack(input)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()

  -- M gradient
  local rows = M:size(M:dim()-1)
  local Mgrad = self.gradInput[1]
  local k_Mrep = k:view(-1,1,k:size(k:dim())):expand(M:size()):clone()
  local knorm_Outrep = self.knorm:view(-1,1):expand(self.output:size(1),self.output:size(2))
  Mgrad:set(k_Mrep)

  -- assume rownorms greater than zero (may be mistake later?)
  Mgrad:add(torch.cmul(
      knorm_Outrep,
      -self.output
    ):cdiv(self.rownorms):view(-1,M:size(M:dim()-1),1):expand(
      M:size(1),
      M:size(2),
      M:size(3)
    ),
    M
  )

  local norm_smooth = torch.cmul(
    self.rownorms,
    knorm_Outrep
  ):add(self.smooth)

  local smooth_out = torch.cdiv(
      gradOutput,
      norm_smooth
  )

  Mgrad:cmul(smooth_out:view(-1,M:size(M:dim()-1),1):expand(
      M:size(1),
      M:size(2),
      M:size(3)
    )
  )

  --[[
  for i = 1, rows do
    if self.rownorms[i] > 0 then
      Mgrad[i]:add(-self.output[i] * self.knorm / self.rownorms[i], M[i])
    end
    Mgrad[i]:mul(gradOutput[i] / (self.rownorms[i] * self.knorm + self.smooth))
  end
  ]]

  -- k gradient
  self.gradInput[2]:set(torch.bmm(
      smooth_out:view(-1,1,gradOutput:size(gradOutput:dim())),
      M
    ):squeeze()
  )

  local scale = torch.cmul(self.output, self.rownorms)
    :cdiv(norm_smooth)
  scale = torch.bmm(
    scale:view(-1,1,self.output:size(self.output:dim())),
    gradOutput:view(-1,gradOutput:size(gradOutput:dim()),1)
  ):squeeze():cdiv(self.knorm)

  self.gradInput[2]:add(
    torch.cmul(
      -scale:view(-1,1):expand(k:size(1),k:size(2)),
      k
    )
  )

  return self.gradInput
end
