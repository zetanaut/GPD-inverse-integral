from spectral import Elements as PolynomialElements

def Elements(xi,degree=6,layout='graded'):
 if layout.startswith('log'):
  from enriched import LogElements
  return LogElements(xi,degree,layout)
 return PolynomialElements(xi,degree,layout)
