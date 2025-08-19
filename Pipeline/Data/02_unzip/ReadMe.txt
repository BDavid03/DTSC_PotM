This dataset contains information about results of Big Bash League T20 Cricket.
1. The main file is: MatchList.csv
   This file contains a row for each match played in the first 7 Men's BBL seasons (2011/2 to 2017/8) and the first 3 Women's BBL seasons (2015/6 to 2017/8).
   Each row contains the following columns:
   ========================================
   League = "BBL" or "WBBL"
   Season = 1 to 7
   Final = Each season has 2 semi-finals and a grand final
   Game_IDno = game ID number used to link to the files containing the ball-by-ball outcome info
   T1 = Name of Team1, the team batting first
   W1 = Wickets lost by Team1
   R1 = Runs scored by Team1
   O1 = Overs completed Team1
   M1 = Maximum possible overs Team1 could have batted 
        [NOTE: So, O1=M1 as long as Team1 did not lose all 10 wickets before M1 overs had been played]
   T2 = Name of Team2, the team batting second
   W2 = Wickets lost by Team2
   R2 = Runs scored by Team2
   O2 = Overs completed Team2
   M2 = Maximum possible overs Team2 could have batted
        [NOTE: So, O2=M2 as long as Team2 did not lose all 10 wickets before M2 overs had been played]
   Win = 1 or 2 if Team1/Team2 win respectively in regulation play (0 if match ended regulation play in a tie, see notes below; blank if match had no result)
   Toss = 1 or 2 if Team1/Team2 won the initial coin toss
   Home = 1 or 2 for Team1/Team2 being the home team (NOTE: 0 means game played at nuetral site or at both teams' home ground)
   PotM = Name of the Player of the Match award winner
   RRD = "Relative Resource Differential", which is a DLS-based measure of margin of victory
         [NOTE: RRD is a value between 0 and 1, larger values corresponding to bigger wins]
   DLSTrgt = DLS Target, if match was interrupted (blank otherwise)
   DLSInfo = Description of the interruption circumstances 

   [NOTES: 
     1. BBL03 match 36552 finished regulation in a tie, Scorchers won in a Super-Over.
     2. BBL04 match 37621 finished regulation in a tie, Stars won in a Super-Over.
     3. BBL06 match 40467 finished regulation in a tie, Sixers won in a Super-Over.
     4. WBBL01 match 39041 finished regulation in a tie, Strikers won in a Super-Over.
     5. WBBL01 match 39025 had 5 penalty runs awarded to the Hurricanes. (Any info you can find as to when and why will be most appreciated!!!)
     6. WBBL02 matches 40501, 40532 and 40540 finished regulation in ties, the Hurricanes, Thunder and Heat, respectively, won in Super-Overs.
     7. During WBBL02, there were two matches (one between Thunder and Strikers in Sydney and one between Strikers and Hurricanes in Adelaide) that were
        abandoned due to rain.  As there was no play at all in these matches, they have been removed from the data and account for the missing values in the
        Game_IDno sequence.
     8. WBBL03 matches 42045, 42057 and 42074 finished regulation in ties, the Strikers, Renegades and Stars, respectively, won in Super-Overs.
     9. Ball-by-ball data for Super-Overs not provided, RRD for matches tied in regulation currently set at 0.]

2. The Ball-by-Ball Outcome Data for each of the 10 Seasons (7 men's and 3 women's): BBL01.csv,BBL02.csv,...,WBBL03.csv
   The files contain a row for each delivery bowled in every match for that season.
   Each row (ie each game delivery) contains the following columns:
   ========================================
   Game_IDno = game ID number used to link to the file containing the overall outcome info:
   Batsman_Hand = Handedness of the batter
   Batsman_Name = Name of the Batsman
   Batsman_Runs = Runs off the bat
   Batsman_Shot = Type of shot attempted
   Batsman_Team = Team of the batter
   Boundary = indicator of whether the ball was struck to (or over) the boundary rope
   Bowler_Hand = Handedness of the bowler
   Bowler_Name = Name of bowler
   Bowler_Type = Kind of bowler (Leg Spin, Off Space, Seam, etc.)
   Bowler_Length = Length of Delivery
   Bowler_Line = Direction of Delivery
   Bowler_Wckt = "Over" or "Around" the wicket
   Bowler_Team = Team of the bowler
   Bye = Number of byes conceded
   Dismissal_Type = Type of Dismissal ("Not Out" if no dismissal)
   Batsman_Dismissed = "Striker" or "Non-Striker"
   Field_Position = Name of fielding position shot went to
   Fielder_Name = Name of primary fielder
   FreeHit = Indicator of whether delivery was a "free hit" (i.e., batsman could not get out)
   Innings = 1 or 2
   LegBye = Number of leg byes
   NoBall = Indicator of whether delivery was a no ball
   NonStriker_Name = Name of non-striking batter
   OverNum = Number of overs bowled so far (so in first over value is 0 and in last over value is 19)
   BallNum = Number of ball in the over (NOTE: Can be higher than 6 if there are illegal deliveries)
   BallNumLgl = Number of legal balls in the over (NOTE: Must be between 1 and 6, only increments when
                                                         a legal delivery has been bowled; i.e., not a
                                                         wide or no ball)
   PPlay = Indicator of whether delivery was in an initial Power Play
   Wide = Number of wide runs conceded

   [NOTES:
     1. In general, bowlers have a unique "type", but some are able to bowl two.  Usually the combination would be Medium Seam and Off Spin.
     2. Per above, Seam bowlers can change their speed category between Fast and Medium over different seasons (which is partly an indication
        that this classification is not well controlled, and so the differences between Fast and Medium categories in this data should be
        taken with a grain of salt.
     3. In theory, a single bowler should deliver all balls in an over; however, occasionally a bowler will be injured and a second bowler will
        finish the remaining deliveries.]
