"""Matched reference-scale networks: identical coordinates, parity and envelope.
No Q2 input. Neither raw neural architecture enforces polynomiality/positivity.
"""
import torch
from torch import nn

class ReferenceNetwork(nn.Module):
 def __init__(self,architecture):
  super().__init__();self.architecture=architecture
  if architecture=='plain':
   self.net=nn.Sequential(nn.Linear(3,42),nn.Tanh(),nn.Linear(42,42),nn.Tanh(),nn.Linear(42,42),nn.Tanh(),nn.Linear(42,3))
  elif architecture=='film':
   width=24;self.input=nn.Linear(1,width);self.blocks=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
   self.conditioners=nn.ModuleList([nn.Sequential(nn.Linear(2,width),nn.Tanh(),nn.Linear(width,2*width)) for _ in range(2)])
   self.output=nn.Linear(width,3)
   for c in self.conditioners:nn.init.normal_(c[-1].weight,std=.01);nn.init.zeros_(c[-1].bias)
  else:raise ValueError(architecture)
 def _raw(self,x,xi):
  z=(xi.square()-.18)/.18
  if self.architecture=='plain':
   return self.net(torch.stack((x,z[:,None].expand_as(x),torch.zeros_like(x)),dim=-1))
  c=torch.stack((z,torch.zeros_like(z)),dim=-1)
  h=torch.tanh(self.input(x[...,None]))
  for b,condition in zip(self.blocks,self.conditioners):
   gamma,beta=condition(c).chunk(2,dim=-1)
   h=h+torch.tanh((1+gamma[:,None,:])*b(h)+beta[:,None,:])/2
  return self.output(h)
 def forward(self,x,xi):
  p=self._raw(x,xi);m=self._raw(-x,xi)
  raw=torch.cat(((p[...,:2]-m[...,:2])/2,(p[...,2:]+m[...,2:])/2),dim=-1)
  return (raw*torch.clamp(1-x.square(),min=0).square()[...,None]).transpose(-1,-2)
