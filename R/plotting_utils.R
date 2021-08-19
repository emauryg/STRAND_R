## Functions to generate plots

match_signatures <- function(mod_test, mod_ref){
    ## Match signatures from model to a reference model
    ## Input:
    ##     mod0: HDmutsig output that you wish to align
    ##     mod_ref: reference model that you wish to align to
    ## Output:
    ##     indexing of the model. 
    
    trinuc_test = mod_test$Bparam$T0$sum(dim=c(1,2))$transpose(1,2)
    
    trinuc_ref = mod_ref$Bparam$T0$sum(dim=c(1,2))$transpose(1,2)
    
    
    # library(clue)
    
    dist_models <- as_array(trinuc_ref$cpu()) %*% t(as_array(trinuc_test$cpu()))
    matching <- solve_LSAP(dist_models, maximum=TRUE)
    
    return(matching)
}

#' Generate plot of trinucleotide contexts
#'
#' @param mod0 is the runEM output
#' @param model_type is the model type i.e. STRAND or TENSIG
#' @return ggplot object with the 96 trinucleotide contexts
#' @export
make_96plot <- function(mod0, model_type="STRAND"){
    if(model_type=="STRAND"){
        T_tensor = tf(T0 = mod0$Bparam$T0$clone(),covs = mod0$Bparam$factors,missing_rate = make_m__(Y = count_matrix))
        # trinuc_probs = mod0$Bparam$T0$sum(dim=c(1,2))
        trinuc_probs = T_tensor$sum(dim=c(1,2,3,4,5))
            signames = paste0("Signature ",1:ncol(trinuc_probs))

        if (cuda_is_available()){
            trinuc_probs = trinuc_probs$cpu()
            trinuc_probs = as_array(trinuc_probs)
        } else {
            trinuc_probs = as_array(trinuc_probs)
        }
        colnames(trinuc_probs) = signames


        p1 = trinuc_probs %>% data.frame() %>% mutate(trinuc = trinuc_subs) %>% 
                        mutate(subs =rep(c("C>A","C>G","C>T","T>A","T>C","T>G"),each=16)) %>%
                        pivot_longer(cols= starts_with("Signature"), names_to="sigs", values_to ="probs") %>%
                        ggplot(aes(x=trinuc, y=probs)) + geom_bar(aes(color= subs, fill=subs),stat="identity",width = 0.8) + 
                            facet_grid(sigs~subs, scales="free_x")+ 
                            theme_classic() + 
                            scale_fill_manual(values = c("dodgerblue3","black","darkred","gray","chartreuse4","orange")) +
                            scale_color_manual(values = c("dodgerblue3","black","darkred","gray","chartreuse4","orange")) +
                            theme(strip.text = element_text(size= 16, face="bold"), axis.text.x = element_text(angle=90))
        p1
        return(p1)
    } else{
       ## For tensorSig we need to bootstrap since there is a lot of variability between the signatures. 
       niter = 30
       trinuc_boots = torch_zeros(c(niter, 96,K))
       D = ncol(mod0$VIparam$E)
       for ( i in 1:niter){
            sample_index = sample(1:D, size=D, replace=TRUE)
            count_boot = count_matrix[..,sample_index]
            X_boot = X_tensor[sample_index]
            mod_tmp = fit_TS(Y=count_boot,X=X_boot, K=K, anno_dims=anno_dims, tau=50)
            T_tensor = calculate_T_ts(mod_tmp)
            trinuc_probs = T_tensor$sum(dim=c(1,2,3,4,5))/T_tensor$sum(dim=c(1,2,3,4,5,6))
            trinuc_boots[i] = trinuc_probs[,match_signatures(mod_tmp, mod0)]
       }

        mid_probs = torch_quantile(trinuc_boots,q = 0.5,dim =1)$squeeze()
        top_probs = torch_quantile(trinuc_boots, q=0.975,dim=1)$squeeze()
        bottom_probs = torch_quantile(trinuc_boots, q=0.125, dim=1)$squeeze()
        signames = paste0("Signature ",1:ncol(trinuc_probs))


        mid_probs = as_array(mid_probs)
        top_probs = as_array(top_probs)
        bottom_probs = as_array(bottom_probs)

        signames = paste0("Signature ",1:ncol(trinuc_probs))
        colnames(mid_probs) = colnames(top_probs) = colnames(bottom_probs) = signames


        p1 = mid_probs %>% as_tibble() %>% mutate(trinuc= trinuc_subs) %>% 
                mutate(subs =rep(c("C>A","C>G","C>T","T>A","T>C","T>G"),each=16)) %>%
                    pivot_longer(cols= starts_with("Signature"), names_to="sigs", values_to ="probs") %>%
                mutate(top_ci = top_probs %>% as_tibble() %>% 
                    pivot_longer(cols= starts_with("Signature"), names_to="sigs", values_to ="probs") %>% pull(probs)) %>%
                mutate(bottom_ci = bottom_probs %>% as_tibble() %>% 
                    pivot_longer(cols= starts_with("Signature"), names_to="sigs", values_to ="probs") %>% pull(probs)) %>%
            ggplot(aes(x=trinuc, y=probs)) + geom_bar(aes(color= subs, fill=subs),stat="identity",width = 0.8) + 
                geom_errorbar(aes(ymin=bottom_ci, ymax=top_ci), width=0.2) +
                facet_grid(sigs~subs, scales="free_x")+ 
                theme_classic() + 
                scale_fill_manual(values = c("dodgerblue3","black","darkred","gray","chartreuse4","orange")) +
                scale_color_manual(values = c("dodgerblue3","black","darkred","gray","chartreuse4","orange")) +
                theme(strip.text = element_text(size= 16, face="bold"), axis.text.x = element_text(angle=90))

        p1
        return(p1)
    }

}


## Plot factors
# library(pheatmap)
#' Plots factor matrices
#'
#' @param mod0 A fitted model object
#' @param model_type The model type (e.g. "STRAND")
#' @return matrices with factor terms that can be plotted using pheatmap
#' @export
plot_factors <- function(mod0, model_type="STRAND"){
    if(model_type=="STRAND"){
        bt = mod0$Bparam$factors$bt$clone()
        br = mod0$Bparam$factors$br$clone()
        epi = mod0$Bparam$factors$epi$clone()
        nuc = mod0$Bparam$factors$nuc$clone()
        clu = mod0$Bparam$factors$clu$clone()

        if(cuda_is_available()){
            bt = as_array(bt$cpu())
            br = as_array(br$cpu())
            epi = as_array(epi$cpu())
            nuc = as_array(nuc$cpu())
            clu = as_array(clu$cpu())
        } else{
            bt = as_array(bt)
            br = as_array(br)
            epi = as_array(epi)
            nuc = as_array(nuc)
            clu = as_array(clu)
        }

        signames = paste0("Signature ", 1:ncol(bt))
        colnames(bt) = colnames(br) = colnames(epi) = colnames(nuc) = colnames(clu) = signames
        rownames(bt) = c("ts_minus","ts_plus")
        rownames(br) = c("rs_minus","rs_plus")
        rownames(epi) = c("epi_none", "active TSS", "flanking active TSS","trasncr. at gene 5' and 3'", "strong transcription","weak transcription","genic enhancers","enhancers",
        "ZNF genes + repeats", "heterochromatin", "bivalent/poised TSS", "flanking bivalent TSS/Enh", "bivalent enhancer","repressed polycomb", "weak repressed polycomb","quiescent/low")
        rownames(nuc) = c("nuc_none","minor_out","minor_in","linker")
        rownames(clu) = c("not_clustered","clustered")

        br_plot = pheatmap(t(br),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "Replication bias", cellheight=80, cellwidth=80)
        bt_plot = pheatmap(t(bt),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "Transcription bias", cellheight=80, cellwidth=80)
        epi_plot = pheatmap(t(epi),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "Epigenetic effect", cellheight=40, cellwidth=40)
        nuc_plot = pheatmap(t(nuc),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "Nucleosome effect", cellheight=80, cellwidth=80)
        clu_plot =pheatmap(t(clu),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "Clustering effect", cellheight=80, cellwidth=80)
        
        return(list(br=br_plot, bt = bt_plot, epi=epi_plot,nuc=nuc_plot, clu=clu_plot))

    } else {
        bt = mod0$Bparam$factors$bt$clone()
        br = mod0$Bparam$factors$br$clone()
        epi = mod0$Bparam$factors$epi$clone()
        nuc = mod0$Bparam$factors$nuc$clone()
        clu = mod0$Bparam$factors$clu$clone()
        if(cuda_is_available()){
            bt = as_array(bt$cpu())
            br = as_array(br$cpu())
            epi = as_array(epi$cpu())
            nuc = as_array(nuc$cpu())
            clu = as_array(clu$cpu())
        } else{
            bt = as_array(bt)
            br = as_array(br)
            epi = as_array(epi)
            nuc = as_array(nuc)
            clu = as_array(clu)
        }
        signames = paste0("Signature ", 1:ncol(epi))
        colnames(epi) = colnames(nuc) = colnames(clu) = signames

        rownames(epi) = c("epi_none", "active TSS", "flanking active TSS","trasncr. at gene 5' and 3'", "strong transcription","weak transcription","genic enhancers","enhancers",
        "ZNF genes + repeats", "heterochromatin", "bivalent/poised TSS", "flanking bivalent TSS/Enh", "bivalent enhancer","repressed polycomb", "weak repressed polycomb","quiescent/low")
        rownames(nuc) = c("nuc_none","minor_out","minor_in","linker")
        rownames(clu) = c("not_clustered","clustered")

        bias_mat = data.frame(bt = bt, br = br) %>% as.matrix()
        bias_mat = t(bias_mat)
        colnames(bias_mat) = signames

        bias_plot = pheatmap(t(bias_mat),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "transcrpition/Replication bias", cellheight=80, cellwidth=80)
        epi_plot = pheatmap(t(epi),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "Epigenetic effect", cellheight=40, cellwidth=40)
        nuc_plot = pheatmap(t(nuc),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "Nucleosome effect", cellheight=80, cellwidth=80)
        clu_plot =pheatmap(t(clu),cluster_rows = FALSE, cluster_cols=FALSE, border_color = "black", fontsize= 16, display_numbers = TRUE,
                fontsize_number = 16, angle_col = 45, number_color="black", main = "Clustering effect", cellheight=80, cellwidth=80)

        return(list(bias_plot=bias_plot, epi=epi_plot,nuc=nuc_plot, clu=clu_plot))

    }

}
