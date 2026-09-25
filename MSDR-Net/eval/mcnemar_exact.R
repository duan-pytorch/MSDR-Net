# Exact two-sided McNemar test for paired classification discordance
# (manuscript Section 2.7; Supplementary Table S2).
#
# Requires the R package exact2x2 (version 1.6.6):
#     install.packages("exact2x2")
#
# Usage:
#     Rscript eval/mcnemar_exact.R predictions_msdr_net_test.csv predictions_convnext_test.csv

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript mcnemar_exact.R <reference.csv> <comparator.csv>")
}

suppressPackageStartupMessages(library(exact2x2))

ref <- read.csv(args[1])
cmp <- read.csv(args[2])
merged <- merge(ref, cmp, by = c("patient_id", "y_true"), suffixes = c("_ref", "_cmp"))
if (nrow(merged) != nrow(ref)) {
  stop("Prediction files could not be aligned by patient_id/y_true.")
}

threshold <- 0.5
correct_ref <- as.integer(merged$y_score_ref >= threshold) == merged$y_true
correct_cmp <- as.integer(merged$y_score_cmp >= threshold) == merged$y_true

# Discordance table: b = reference only, c = comparator only.
b <- sum(correct_ref & !correct_cmp)
c <- sum(!correct_ref & correct_cmp)

tab <- matrix(c(sum(correct_ref & correct_cmp), b,
                c, sum(!correct_ref & !correct_cmp)), nrow = 2, byrow = TRUE)
result <- mcnemar.exact(tab)

cat("Comparison:", args[1], "vs", args[2], "\n")
cat("b (reference only) =", b, ", c (comparator only) =", c, "\n")
cat("Exact two-sided McNemar p =", result$p.value, "\n")
