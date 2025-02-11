bsub -q gsla-cpu -R rusage[mem=4200] -J yoyo /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 32  --genome_size 128 --beta 0.5 --rho 0.25 --mating_strategy all_vs_all

bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 14 --genome_size 100 --beta 0.5 --rho 0.25 --mating_strategy all_vs_all --output_dir /home/labs/pilpel/barc/sexy_yeast/14gen_100genomes_0.5beta_0.25rho_all_vs_all

bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 14 --genome_size 128 --beta 0.5 --rho 0.25 --mating_strategy mating_types --output_dir /home/labs/pilpel/barc/sexy_yeast/14gen_128genomes_0.5beta_0.25rho_mating_types

bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 8 --genome_size 128 --beta 0.5 --rho 0.25 --mating_strategy mating_types --output_dir /home/labs/pilpel/barc/sexy_yeast/8gen_128genomes_0.5beta_0.25rho_mating_types

bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 8 --genome_size 128 --beta 0.5 --rho 0.25 --mating_strategy mating_types --output_dir /home/labs/pilpel/barc/sexy_yeast/8gen_128genomes_0.5beta_0.25rho_mating_types


bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 8 --genome_size 128 --beta 0.3 --rho 0.25 --mating_strategy mating_types --output_dir /home/labs/pilpel/barc/sexy_yeast/8gen_128genomes_0.3beta_0.25rho_mating_types

bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 8 --genome_size 128 --beta 0.3 --rho 0.25 --mating_strategy all_vs_all --output_dir /home/labs/pilpel/barc/sexy_yeast/8gen_128genomes_0.3beta_0.25rho_mating_types