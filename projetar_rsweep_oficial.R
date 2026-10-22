#!/usr/bin/env Rscript
# ==============================================================================
# Script Oficial de Projeção SWeeP para Matrizes Celulares (rSWeeP Bioconductor)
# ==============================================================================
# Este script implementa a execução oficial do método SWeeP (Spaced Words Projection)
# utilizando o pacote homologado rSWeeP (v1.18+ / AIBIA Lab / UFPR) para redução de
# dimensionalidade de matrizes transcriptômicas de expressão celular (scRNA-seq).
# ==============================================================================

suppressPackageStartupMessages({
    library(Matrix)
    library(rSWeeP)
    library(RcppCNPy)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
    cat("Uso: Rscript projetar_rsweep_oficial.R <mtx_in> <orthbase_path> <out_proj_path> <dim_proj> <seed> [clean_mtx=TRUE]\n")
    quit(status = 1)
}

mtx_in <- args[1]
orthbase_path <- args[2]
out_proj_path <- args[3]
dim_proj <- as.integer(args[4])
seed_val <- as.integer(args[5])
clean_mtx <- ifelse(length(args) >= 6, as.logical(args[6]), TRUE)

t_inicio <- proc.time()

cat("====================================================================\n")
cat("      PROJEÇÃO OFICIAL VIA PACOTE rSWeeP (UFPR / Bioconductor)      \n")
cat("====================================================================\n")
cat(sprintf("• Arquivo MTX de entrada : %s\n", mtx_in))
cat(sprintf("• Caminho da Ortobase    : %s\n", orthbase_path))
cat(sprintf("• Saída Projetada (.npy) : %s\n", out_proj_path))
cat(sprintf("• Dimensão Alvo (v)      : %d\n", dim_proj))
cat(sprintf("• Semente Pseudoaleatória: %d\n", seed_val))
cat("====================================================================\n\n")

# 1. Carregamento da Matriz Esparsa
cat("1. Lendo matriz esparsa via Matrix::readMM...\n")
t_mtx <- proc.time()
mat_esparsa <- Matrix::readMM(mtx_in)
mat_esparsa <- as(mat_esparsa, "CsparseMatrix")
n_celulas <- nrow(mat_esparsa)
n_genes <- ncol(mat_esparsa)
cat(sprintf("   ✓ Matriz carregada em %.2fs (%d células × %d genes)\n\n",
            (proc.time() - t_mtx)[[3]], n_celulas, n_genes))

# 2. Resolução da Ortobase Oficial
cat("2. Resolvendo Ortobase Oficial SWeeP...\n")
orthbase_rds <- sub("\\.npy$", ".rds", orthbase_path)

if (file.exists(orthbase_rds)) {
    cat(sprintf("   • Carregando ortobase congelada existente (.rds): %s\n", orthbase_rds))
    base_obj <- readRDS(orthbase_rds)
    if (nrow(base_obj$mat) != n_genes || ncol(base_obj$mat) != dim_proj) {
        stop(sprintf("Incompatibilidade dimensional na ortobase: esperado (%d, %d), encontrado (%d, %d)",
                     n_genes, dim_proj, nrow(base_obj$mat), ncol(base_obj$mat)))
    }
} else if (file.exists(orthbase_path)) {
    cat(sprintf("   • Carregando ortobase congelada existente (.npy): %s\n", orthbase_path))
    mat_npy <- RcppCNPy::npyLoad(orthbase_path)
    if (nrow(mat_npy) != n_genes || ncol(mat_npy) != dim_proj) {
        stop(sprintf("Incompatibilidade dimensional na ortobase: esperado (%d, %d), encontrado (%d, %d)",
                     n_genes, dim_proj, nrow(mat_npy), ncol(mat_npy)))
    }
    base_obj <- list(mat = mat_npy, seed = seed_val, version = "SWeeP v2.9")
    # Salva também a cópia .rds para otimizar futuras leituras no ambiente R
    saveRDS(base_obj, orthbase_rds)
} else {
    cat(sprintf("   • Gerando nova ortobase canônica via rSWeeP::orthBase (lin=%d, col=%d, seed=%d)...\n",
                n_genes, dim_proj, seed_val))
    t_base <- proc.time()
    base_obj <- rSWeeP::orthBase(lin = n_genes, col = dim_proj, seed = seed_val)
    cat(sprintf("   ✓ Ortobase gerada em %.2fs!\n", (proc.time() - t_base)[[3]]))
    
    # Persistência da base em RDS nativo e NPY para uso em Python
    saveRDS(base_obj, orthbase_rds)
    RcppCNPy::npySave(orthbase_path, base_obj$mat)
    cat(sprintf("   ✓ Ortobase persistida com sucesso em: %s e %s\n", orthbase_rds, orthbase_path))
}

# 3. Execução da Projeção SWeeP Oficial
cat("\n3. Executando projeção espectral via função homologada rSWeeP::SWeeP...\n")
t_proj <- proc.time()
res_sweep <- rSWeeP::SWeeP(input = mat_esparsa, orthbase = base_obj, RNAseqdata = FALSE)
tempo_proj <- (proc.time() - t_proj)[[3]]

proj_matriz <- as.matrix(res_sweep$proj)
cat(sprintf("   ✓ Projeção concluída com sucesso em %.2fs!\n", tempo_proj))
cat(sprintf("   • Dimensão resultante: %d células × %d coordenadas\n",
            nrow(proj_matriz), ncol(proj_matriz)))

# 4. Exportação Binária em Formato NumPy (.npy)
cat(sprintf("\n4. Salvando projeção final em formato NumPy (.npy): %s\n", out_proj_path))
RcppCNPy::npySave(out_proj_path, proj_matriz)
cat("   ✓ Matriz projetada gravada com precisão float nativa!\n")

# 5. Limpeza de Arquivo Temporário MTX (caso solicitado)
if (clean_mtx && file.exists(mtx_in)) {
    unlink(mtx_in)
    cat(sprintf("   ✓ Arquivo intermediário temporário removido: %s\n", mtx_in))
}

tempo_total <- (proc.time() - t_inicio)[[3]]
cat(sprintf("\n🎯 Pipeline oficial rSWeeP finalizado com êxito absoluto em %.2fs!\n", tempo_total))
cat("====================================================================\n")
