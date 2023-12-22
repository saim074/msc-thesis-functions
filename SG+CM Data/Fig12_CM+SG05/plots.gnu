
reset
set term pdf 
set out 'plots.pdf'
#
set xrange [-11:11]
set yrange [-500:500]
set xtics 2
set ytics 100
#set zeroaxis
#
set key bottom
set xlabel 'Displacement [mm]'
set ylabel 'Shear Force [N]'
#
plot 'exp_4-smooth.dat'  u 2:1 tit ' 4 mm/min' w l lt 1 lw 3,\
     'exp_40-smooth.dat' u 2:1 tit '40 mm/min' w l lt 3 lw 3
#
#
#
set xtics 0.2
set ytics 0.2
#
set xrange [0.6:2.2]
set yrange [-1.2:1.2]
#
set xlabel 'Stretch [-]'
set ylabel 'Nominal Stress [MPa]'
#
plot 'relax.dat' u ($3/100+1):1 tit 'Discontinuous Uniaxial' w l lt 1 lw 3

quit
