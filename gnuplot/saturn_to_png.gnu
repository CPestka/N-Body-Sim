reset

set terminal pngcairo size 1920,1080 enhanced font 'Verdana,10'

set border 0
unset tics
range=3.95e+8
set xrange [-range:range]
set yrange [-range:range]
set zrange [-range:range]
system('mkdir -p ../data/png/saturn')


n=0
num_big_steps = 100;
do for [ii=1:num_big_steps] {
    set output sprintf('../data/png/saturn/nBody%03.0f.png',n)
    plotline = 'splot'
    splot 'N_Body_Timestep_0.txt' u 2:3:4 every ::1::ii w l ls 1 lc 1 title "Earth", \
          'N_Body_Timestep_0.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 1 notitle, \
    n=n+1
}
