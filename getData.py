# Name: CHAO SHUOTAN
# Student ID: 22324118
# Programming Time: 28/11/2022 01:19
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType

client.players_advanced_season_totals(
    season_end_year=2022,
    output_type=OutputType.CSV,
    output_file_path="2022_player.csv"
)