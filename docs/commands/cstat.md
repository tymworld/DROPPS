# dps cstat

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate residue-level statistics from a precomputed contact map.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps cstat [-h] [-db] -m FILE -s FILE [-r REFERENCE_GROUP]
                 [-sg SELECTION_GROUP] [-n FILE] -o FILE [-gr]
                 [-gs {HPST,AHCP,HCP,HPST_SC,AHCP_SC,HCP_SC}] [-g FILE]
                 [-ag {sum,average}]

Calculate residue-level statistics from a precomputed contact map.

Known grouping schemes:
  HPST: Aromatic (H,F,W,Y), Hydrophobic (A,I,L,M,V), Other (C,G,P), Charged (K,R,D,E), Polar (N,Q,S,T)
  AHCP: Aromatic (H,F,W,Y), Hydrophobic (A,I,L,M,V,P), Charged (K,R,D,E), Polar (N,Q,S,T,C,G)
  HCP: Hydrophobic (H,F,W,Y,A,I,L,M,V,P), Charged (K,R,D,E), Polar (N,Q,S,T,C,G)
  HPST_SC: Aromatic (H,F,W,Y), Hydrophobic (A,I,L,M,V), Other (C,G,P), Charged+ (K,R), Charged- (D,E), Polar (N,Q,S,T)
  AHCP_SC: Aromatic (H,F,W,Y), Hydrophobic (A,I,L,M,V,P), Charged+ (K,R), Charged- (D,E), Polar (N,Q,S,T,C,G)
  HCP_SC: Hydrophobic (H,F,W,Y,A,I,L,M,V,P), Charged+ (K,R), Charged- (D,E), Polar (N,Q,S,T,C,G)

Input:
  -m FILE, --map-input FILE
                        Input contact-map matrix file (.dat). (required)
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the system
                        topology. (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)
  -g FILE, --group-file FILE
                        Optional custom grouping file with GROUP_NAME
                        ABBREVIATION RESIDUE... on each line. (default: None)

Output:
  -o FILE, --output FILE
                        Output Excel workbook (.xlsx); the extension is added
                        if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -r REFERENCE_GROUP, --reference-group REFERENCE_GROUP
                        Reference index group corresponding to the contact-map
                        x axis. (default: None)
  -sg SELECTION_GROUP, --selection-group SELECTION_GROUP
                        Selection index group corresponding to the contact-map
                        y axis. (default: None)
  -gr, --group-residue  Aggregate residues into residue-type groups. (default:
                        False)
  -gs {HPST,AHCP,HCP,HPST_SC,AHCP_SC,HCP_SC}, --group-scheme {HPST,AHCP,HCP,HPST_SC,AHCP_SC,HCP_SC}
                        Predefined residue-grouping scheme. (default: None)
  -ag {sum,average}, --aggregate {sum,average}
                        Aggregation method for grouped contact statistics.
                        (default: sum)

Notes:
  Summarize a precomputed contact map using the matching TPR and reference/selection group layout. Output is an Excel workbook.

  Use the same group order and averaging convention as the command that generated the map; a matrix of the right size alone does not establish matching residue identities.

Examples:
  dps cstat -m contacts.dat -s run.tpr -o contact-statistics.xlsx
```
