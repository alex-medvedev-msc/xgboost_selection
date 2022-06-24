args <- commandArgs(trailingOnly=T)
trait_name <- args[[1]]
n_snps <- args[[2]]

config <- config::get(file = "conf/config.yml")

configs <- list(
                results.dir=paste(config$results_dir, trait_name,'_', n_snps, sep=''),
                save=T,
                mem=300000,
                use.glmnetPlus=T,
                nCores=12
                )

library('snpnet')

header <- read.csv(file=paste0(config$phenotype_dir,trait_name,"_cov.csv"),nrows=1,sep="\t")
cov <- colnames(header)[5:length(header)-1] 
print(cov)

fit <- snpnet::snpnet(paste0(config$genotype_dir, trait_name,'/chrMAIN_top', n_snps, 'k'),
                      paste0(config$phenotype_dir,trait_name,'_cov.csv'),
                      phenotype=trait_name,
                      split.col='split', 
                      covariates=cov,
                      configs=configs)

preds <- snpnet::predict_snpnet(fit,
                                new_genotype_file=paste0(config$genotype_dir, n_snps, 'k'),
                                new_phenotype_file=paste0(config$phenotype_dir,trait_name,'_cov.csv'),
                                phenotype=trait_name,
                                covariate_names=cov,
                                split_col='split', split_name=c('train', 'val', 'test'))

write(paste(c(n_snps, preds$metric$train[length(preds$metric$val)],preds$metric$val[length(preds$metric$val)], preds$metric$test[length(preds$metric$val)]), sep=','), paste0(trait_name, '_cov.csv'), append=T)


write.csv(preds$prediction$test[,dim(preds$prediction$test)[2]], paste0(config$predictions_dir, trait_name,'_',n_snps,'k_cov_test.csv'), row.names=T)
write.csv(preds$prediction$train[,dim(preds$prediction$train)[2]], paste0(config$predictions_dir, trait_name,'_',n_snps,'k_cov_train.csv'), row.names=T)
write.csv(preds$prediction$val[,dim(preds$prediction$val)[2]], paste0(config$predictions_dir, trait_name,'_',n_snps,'k_cov_val.csv'), row.names=T)
