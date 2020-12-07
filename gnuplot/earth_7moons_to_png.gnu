reset

set terminal pngcairo size 1920,1080 enhanced font 'Verdana,10'

set border 0
unset tics
range=5.0e+8
set xrange [-range:range]
set yrange [-range:range]
set zrange [-range:range]
system('mkdir -p ../data/png/earth7Moons')


n=0
num_big_steps = 100;
do for [ii=1:num_big_steps] {
    n=n+1
    set output sprintf('../data/png/earth7Moons/nBody%03.0f.png',n)
    splot 'N_Body_Particle_0.txt' u 2:3:4 every ::1::ii w l ls 1 lc 1 title "Earth", \
          'N_Body_Particle_0.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 1 notitle, \
          'N_Body_Particle_1.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 2 title "Moon 1", \
          'N_Body_Particle_1.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 2 notitle, \
          'N_Body_Particle_2.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 3 title "Moon 2", \
          'N_Body_Particle_2.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 3 notitle, \
          'N_Body_Particle_3.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 4 title "Moon 3", \
          'N_Body_Particle_3.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 4 notitle, \
          'N_Body_Particle_4.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 5 title "Moon 4", \
          'N_Body_Particle_4.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 5 notitle, \
          'N_Body_Particle_5.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 6 title "Moon 5", \
          'N_Body_Particle_5.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 6 notitle, \
          'N_Body_Particle_6.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 7 title "Moon 6", \
          'N_Body_Particle_6.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 7 notitle, \
          'N_Body_Particle_7.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 8 title "Moon 7", \
          'N_Body_Particle_7.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 8 notitle, \
}
