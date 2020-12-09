reset

set terminal pngcairo size 1920,1080 enhanced font 'Verdana,10'

set border 0
unset tics
range=7.0e+8
set xrange [-range:range]
set yrange [-range:range]
set zrange [-range:range]
system('mkdir -p ../data/png/earth7Moons')


n=0
num_big_steps = 501;
do for [ii=1:num_big_steps] {
    n=n+1
    set output sprintf('../data/png/earth7Moons/nBody%03.0f.png',n)
    splot '../data/raw/earth7Moons/N_Body_Particle_00000.txt' u 2:3:4 every ::1::ii w l ls 1 lc 1 title "Earth", \
          '../data/raw/earth7Moons/N_Body_Particle_00000.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 1 notitle, \
          '../data/raw/earth7Moons/N_Body_Particle_00001.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 2 title "Moon 1", \
          '../data/raw/earth7Moons/N_Body_Particle_00001.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 2 notitle, \
          '../data/raw/earth7Moons/N_Body_Particle_00002.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 3 title "Moon 2", \
          '../data/raw/earth7Moons/N_Body_Particle_00002.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 3 notitle, \
          '../data/raw/earth7Moons/N_Body_Particle_00003.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 4 title "Moon 3", \
          '../data/raw/earth7Moons/N_Body_Particle_00003.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 4 notitle, \
          '../data/raw/earth7Moons/N_Body_Particle_00004.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 5 title "Moon 4", \
          '../data/raw/earth7Moons/N_Body_Particle_00004.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 5 notitle, \
          '../data/raw/earth7Moons/N_Body_Particle_00005.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 6 title "Moon 5", \
          '../data/raw/earth7Moons/N_Body_Particle_00005.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 6 notitle, \
          '../data/raw/earth7Moons/N_Body_Particle_00006.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 7 title "Moon 6", \
          '../data/raw/earth7Moons/N_Body_Particle_00006.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 7 notitle, \
          '../data/raw/earth7Moons/N_Body_Particle_00007.txt' u 2:3:4 every ::1::ii  w l ls 1 lc 8 title "Moon 7", \
          '../data/raw/earth7Moons/N_Body_Particle_00007.txt' u 2:3:4 every ::ii::ii  w p ls 1 lc 8 notitle, \
}
