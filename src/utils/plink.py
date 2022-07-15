import subprocess
import pandas
import os


def run_plink(args):
    completed = subprocess.run(['plink2'] + args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if completed.returncode != 0:
        print(completed.returncode)
        print(str(completed.stderr))
        print(str(completed.stdout))
        
        raise RuntimeError(f'plink2 returned {completed.returncode}, error: {str(completed.stderr)}')
        
def run_plink19(args):
    completed = subprocess.run(['plink'] + args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if completed.returncode != 0:
        print(completed.returncode)
        print(str(completed.stderr))
        print(str(completed.stdout))
        
        raise RuntimeError(f'plink returned {completed.returncode}, error: {str(completed.stderr)}')
    else:
        print(str(completed.stdout))
    
    
def run_flashpca(args):

    with subprocess.Popen(['ukb_ml/tools/flashpca'] + args, stdout=subprocess.PIPE, bufsize=1) as sp:
        for line in sp.stdout:
            print(line, flush=True)

    if sp.returncode != 0:
        print(sp.returncode)
        print(str(sp.stderr))
        
        raise RuntimeError(f'flashpca returned {sp.returncode}, error: {str(sp.stderr)}')

        
def prepare_snps_file(info_path, maf, info, output_file):
    data = pandas.read_csv(info_path, header=None, names=['snp', 'rsid', 'position', 'ref', 'alt', 'maf', 'alt2', 'info'], sep='\t')
    interesting = data[(data.maf > maf) & (data['info'] > info)]
    ids = interesting.rsid.tolist()
    with open(output_file, 'w') as out:
        for _id in ids:
            out.write(_id + '\n')
    return len(ids)


def update_snp_ids(pgen_path):

    if os.path.exists(f'{pgen_path}.backup.pvar'):
        # IDS were already updated, so we just read old ids and update them anyway - idempotent operation
        pvar = pandas.read_csv(f'{pgen_path}.backup.pvar', sep='\t')
    else:
        pvar = pandas.read_csv(f'{pgen_path}.pvar', sep='\t')

    old_pvar = pvar.copy()
    
    pvar.ID = pvar['#CHROM'].astype(str) + ':' + pvar.POS.astype(str) + '_' + pvar.ALT + '_' + pvar.REF + '_' + pvar.POS.astype(str)
    
    old_pvar.to_csv(f'{pgen_path}.backup.pvar', index=False, sep='\t')
    pvar.to_csv(f'{pgen_path}.pvar', index=False, sep='\t')
         