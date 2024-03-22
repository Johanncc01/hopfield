import cProfile
import pstats
from main import main
cProfile.run("main()","restats")
p = pstats.Stats('restats')
p.sort_stats('cumulative').print_stats(20)

