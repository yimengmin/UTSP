
// Jump to a new state by randomly generating a solution
void Jump_To_Random_State()
{
	Generate_Initial_Solution(); 	
}

Distance_Type Markov_Decision_Process(int Inst_Index)
{
	MCTS_Init(Inst_Index);                      // Initialize MCTS parameters
	Generate_Initial_Solution();      // State initialization of MDP
	Local_Search_by_2Opt_Move();	  // 2-opt based local search within small neighborhood	
	MCTS();		                      // Tageted sampling via MCTS within enlarged neighborhood

	// Repeat the following process until termination
    int jump = 0;
	while(((double)clock()-Current_Instance_Begin_Time) /CLOCKS_PER_SEC<Param_T*Virtual_City_Num)
	{
        // Here we restart the max depth, and rec_only
        if (restart) {
            int old_rec_only = rec_only;
            if (restart_reconly) {
                rec_only = rand() % 2;
                if (rec_only != old_rec_only) {
                    Identify_Candidate_Set();
                }
            }
            // Random 
            //Max_Depth = 100 + (rand() % 60);
            // Random v2
            Max_Depth = 10 + (rand() % 80);
        }
        ++jump;
		Jump_To_Random_State();		
		Local_Search_by_2Opt_Move();		
		MCTS();
	}
    cout << "jump:" << jump <<endl;
	
	// Copy information of the best found solution (stored in Struct_Node *Best_All_Node ) to Struct_Node *All_Node 
	Restore_Best_Solution();
	
	if(Check_Solution_Feasible())
		return Get_Solution_Total_Distance();
	else
		return Inf_Cost;
}



