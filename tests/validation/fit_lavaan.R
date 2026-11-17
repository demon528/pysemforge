# Reference fit for the pysemforge validation suite.
#
# Usage:
#   Rscript fit_lavaan.R <input_csv> <output_json>
#
# The CSV must contain columns x1..x6.  The script fits exactly the model that
# the Python test fits -- two correlated factors, three indicators each, with
# the factor variances fixed to 1 (std.lv = TRUE) -- and writes the statistics
# and the standardised loadings so that the two fits can be compared term by
# term.

library(lavaan)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript fit_lavaan.R <input.csv> <output.json>")

d <- read.csv(args[1])

syntax <- "
  F1 =~ x1 + x2 + x3
  F2 =~ x4 + x5 + x6
"
fit <- cfa(syntax, data = d, std.lv = TRUE)
fm <- fitMeasures(fit, c("chisq", "df", "cfi", "tli", "rmsea", "srmr"))

sl <- standardizedSolution(fit)
load <- sl$est.std[sl$op == "=~"]
names(load) <- sl$rhs[sl$op == "=~"]
corr <- sl$est.std[sl$op == "~~" & sl$lhs == "F1" & sl$rhs == "F2"]

out <- list(
  chisq    = unname(fm[["chisq"]]),
  df       = unname(fm[["df"]]),
  cfi      = unname(fm[["cfi"]]),
  tli      = unname(fm[["tli"]]),
  rmsea    = unname(fm[["rmsea"]]),
  srmr     = unname(fm[["srmr"]]),
  loadings = as.list(load),
  factor_corr = unname(corr)
)

writeLines(toJSON(out, auto_unbox = TRUE, digits = 10), args[2])
cat("lavaan reference written to", args[2], "\n")
