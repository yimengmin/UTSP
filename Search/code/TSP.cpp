#include "include/TSP_IO.h"
#include "include/TSP_Basic_Functions.h"
#include "include/TSP_Init.h"
#include "include/TSP_2Opt.h"
#include "include/TSP_MCTS.h"
#include "include/TSP_Markov_Decision.h"
#include "include/TSP_sym.h"
#include <algorithm> // For std::find

// For TSP20-50-100 instances
void Solve_One_Instance(int Inst_Index)
{	
	Current_Instance_Begin_Time=(double)clock();  
	Current_Instance_Best_Distance=Inf_Cost;   	   
	
	// Input			
    cout << "Start Fetch" << endl;
	Fetch_Stored_Instance_Info(Inst_Index);	
	
    cout << "Start Preprocess" << endl;
	//Pre-processing	
	Calculate_All_Pair_Distance();	 	
  	Identify_Candidate_Set();    
	  
    cout << "Start MDP Search" << endl;
	//Search by MDP  	 		  		    
	Markov_Decision_Process(Inst_Index);
			
	double Stored_Solution_Double_Distance=Get_Stored_Solution_Double_Distance(Inst_Index);
	double Current_Solution_Double_Distance=Get_Current_Solution_Double_Distance();
			
	if(Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate > 0.000001)
		Beat_Best_Known_Times++;	
	else if(Current_Solution_Double_Distance/Magnify_Rate-Stored_Solution_Double_Distance/Magnify_Rate > 0.000001)
		Miss_Best_Known_Times++;
	else
		Match_Best_Known_Times++;	
			
	Sum_Opt_Distance+=Stored_Solution_Double_Distance/Magnify_Rate;
	Sum_My_Distance+=Current_Solution_Double_Distance/Magnify_Rate;	
	Sum_Gap += (Current_Solution_Double_Distance-Stored_Solution_Double_Distance)/Stored_Solution_Double_Distance;
		
	printf("\nInst_Index:%d Concorde Distance:%f, MCTS Distance:%f Improve:%f Time:%.2f Seconds\n", Inst_Index+1, Stored_Solution_Double_Distance/Magnify_Rate, 
			Current_Solution_Double_Distance/Magnify_Rate, Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate, ((double)clock()-Current_Instance_Begin_Time)/CLOCKS_PER_SEC);
			
	FILE *fp;   
	fp=fopen(Statistics_File_Name, "a+");     
	fprintf(fp,"\nInst_Index:%d \t City_Num:%d \t Concorde:%f \t MCTS:%f Improve:%f \t Time:%.2f Seconds\n",Inst_Index+1, Virtual_City_Num, Stored_Solution_Double_Distance/1000000,
			Current_Solution_Double_Distance/Magnify_Rate, Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate, ((double)clock()-Current_Instance_Begin_Time)/CLOCKS_PER_SEC); 
	
	fprintf(fp,"Solution: ");
	int Cur_City=Start_City;
	do
	{
		fprintf(fp,"%d ",Cur_City+1);
		Cur_City=All_Node[Cur_City].Next_City;		
	}while(Cur_City != Null && Cur_City != Start_City);
	
	fprintf(fp,"\n"); 
	fclose(fp); 	
			
	Release_Memory(Virtual_City_Num);	
}
 
bool Solve_Instances_In_Batch()
{ 
	ifstream FIC;
	FIC.open(Input_File_Name);  
  
	if(FIC.fail())
	{
    	cout << "\n\nError! Fail to open file"<<Input_File_Name<<endl;
    	getchar();
    	return false;     
	}
  	else
    	cout << "\n\nBegin to read instances information from "<<Input_File_Name<<endl;
    	    
  			 
 	double Temp_X;
 	double Temp_Y;
 	int Temp_City;
    int Rec_Index;
    double Rec_Value;
 	char Temp_String[100]; 	

    cout << "Total Instance Considered: " << Total_Instance_Num << endl;
 	
  	for(int i=0;i<Total_Instance_Num;i++)   
  	{
  		for(int j=0;j<Temp_City_Num;j++)
  		{
			FIC>>Temp_X;
			FIC>>Temp_Y;
			Stored_Coordinates_X[i][j]=Temp_X;
			Stored_Coordinates_Y[i][j]=Temp_Y;			
		}
		
		FIC>>&Temp_String[0];  
		for(int j=0;j<Temp_City_Num;j++)
  		{
			FIC>>Temp_City;
			Stored_Opt_Solution[i][j]=Temp_City-1;					
		}  	
		
		FIC>>Temp_City;			
        // Here we start reading the recomend cities
        FIC >> &Temp_String[0];
        for (int j = 0; j < Temp_City_Num; ++j) {
            for (int k = 0; k < Rec_Num; ++k) {
                FIC >> Rec_Index;
                Sparse_Stored_Rec[i][j].push_back(Rec_Index - 1);
            }
        }
        FIC >> &Temp_String[0];
        for (int j = 0; j < Temp_City_Num; ++j) {
            for (int k = 0; k < Rec_Num; ++k) {
                FIC >> Rec_Value;
                Sparse_Stored_Rec_Value[i][j].push_back(Rec_Value);
            }
        }

//  	cout <<"\nRead instances finished. Begin to search."<<endl;
        for (int j = 0; j < Temp_City_Num; ++j) {
		for(int k = 0; k < Temp_City_Num; ++k)
		{
		Stored_Rec_Value[i][j].push_back(0.0);
		}
            }

	for (int j = 0; j < Temp_City_Num; ++j) {
	    for (int l = 0; l < Temp_City_Num; ++l) {
	        // Use std::find to look for 'l' in Sparse_Stored_Rec[i][j]
	        auto it = std::find(Sparse_Stored_Rec[i][j].begin(), Sparse_Stored_Rec[i][j].end(), l);
	        if (it != Sparse_Stored_Rec[i][j].end()) {
	            // If found, calculate the index using std::distance
	            int index = std::distance(Sparse_Stored_Rec[i][j].begin(), it);
	            //std::cout << l << " index: " << index << std::endl;
		    Stored_Rec_Value[i][j][l] =  Sparse_Stored_Rec_Value[i][j][index];
	        }
	    }
	}


	// H' = H + H^T
	symmetrizeMatrix(Stored_Rec_Value[i], Max_City_Num);

	for (int j = 0; j < Temp_City_Num; ++j) {
		for (int m = 0; m < Temp_City_Num; ++m)
               	Stored_Rec[i][j].push_back(m);
	}



	
	}      
  	FIC.close();  


    cout << "Inst Num Per Batch " << Inst_Num_Per_Batch << endl;	
	if((Index_In_Batch+1)*Inst_Num_Per_Batch < Total_Instance_Num)
		Test_Inst_Num=Inst_Num_Per_Batch;
	else
		Test_Inst_Num=Total_Instance_Num-Index_In_Batch*Inst_Num_Per_Batch; 
	cout<<"\nNumber of instances in current batch: " <<Test_Inst_Num <<endl; 
	
	FILE *fp;   
	fp=fopen(Statistics_File_Name, "w+");     
	fprintf(fp,"Number_of_Instances_In_Current_Batch: %d\n",Test_Inst_Num);  
	fclose(fp);   
	
			
  	for(int i=Index_In_Batch*Inst_Num_Per_Batch;i<(Index_In_Batch+1)*Inst_Num_Per_Batch && i<Total_Instance_Num;i++)	   
		Solve_One_Instance(i);	  
        
  	return true;  
}

int main(int argc, char ** argv)
{  	
	double Overall_Begin_Time=(double)clock();
	
    //srand(Random_Seed); 	
    srand(time(NULL));

	Index_In_Batch=atoi(argv[1]);
	Statistics_File_Name=argv[2];
	Input_File_Name=argv[3];
	Temp_City_Num=atoi(argv[4]);
    use_rec = atoi(argv[5]);
    rec_only = atoi(argv[6]);

    Max_Candidate_Num = atoi(argv[7]);
    Max_Depth = atoi(argv[8]);
    Alpha = atof(argv[9]);
    Beta = atof(argv[10]);
    Param_H = atof(argv[11]);
    restart = atoi(argv[12]);
    restart_reconly = atoi(argv[13]);

    cout << "record some exp parameters here: !!" << endl;
    cout << "Alpha: " << Alpha << endl;
    cout << "Beta: " << Beta << endl;
    cout << "Param_H: " << Param_H << endl;
    cout << "Param_T: " << Param_T << endl;
    cout << "#Candidate Set: " << Max_Candidate_Num << endl;
    cout << "Max Depth " << Max_Depth << endl;
    cout << "rec_only " << rec_only << endl;
    cout << "restart" << restart << endl;
    cout << "restart_reconly" << restart_reconly << endl;

	Solve_Instances_In_Batch(); 
  	
	FILE *fp;    	  
	fp=fopen(Statistics_File_Name, "a+"); 
	fprintf(fp,"\n\nIndex_In_Batch: %d, Avg_Concorde_Distance: %f Avg_MCTS_Distance: %f Avg_Gap: %f Total_Time: %.2f Seconds \n Beat_Best_Known_Times: %d Match_Best_Known_Times: %d Miss_Best_Known_Times: %d \n",
			Index_In_Batch, Sum_Opt_Distance/Test_Inst_Num,Sum_My_Distance/Test_Inst_Num, Sum_Gap/Test_Inst_Num, ((double)clock()-Overall_Begin_Time)/CLOCKS_PER_SEC, Beat_Best_Known_Times, Match_Best_Known_Times, Miss_Best_Known_Times);
	fclose(fp);
	
	printf("\n\nIndex_In_Batch: %d, Avg_Concorde_Distance: %f Avg_MCTS_Distance: %f Avg_Gap: %f Total_Time: %.2f Seconds \n Beat_Best_Known_Times: %d Match_Best_Known_Times: %d Miss_Best_Known_Times: %d \n",
			Index_In_Batch, Sum_Opt_Distance/Test_Inst_Num,Sum_My_Distance/Test_Inst_Num, Sum_Gap/Test_Inst_Num, ((double)clock()-Overall_Begin_Time)/CLOCKS_PER_SEC, Beat_Best_Known_Times, Match_Best_Known_Times, Miss_Best_Known_Times);
	getchar();

	return 0;
}


/* 
// For TSPLib instances
int Solve_One_Instance()
{	
	Current_Instance_Begin_Time=(double)clock();  
	Current_Instance_Best_Distance=Inf_Cost;    
				
	Read_Instance_Info(Input_Inst_Name);	
	
	//Pre-processing
	Calculate_All_Pair_Distance();	 	
  	Identify_Candidate_Set();   
	
	//Search by MDP   			  		    
	Markov_Decision_Process();
	
	Current_Instance_Best_Distance=Get_Solution_Total_Distance();

	if(Current_Instance_Best_Distance == Best_Known_Result)
		Match_Best_Known_Times++;
	else
		Miss_Best_Known_Times++;
		
	Sum_Gap += (double)(Current_Instance_Best_Distance-Best_Known_Result)/Best_Known_Result;	
	
		
	FILE *fp;   
	fp=fopen(Statistics_File_Name, "a+");     
	fprintf(fp,"\n%s \t City Num: %d \t Best_known:%d \t MCTS: %d \t Gap: %d \t Time:%.2f Seconds\n",	Input_Inst_Name, Virtual_City_Num, Best_Known_Result,Current_Instance_Best_Distance, 
			Current_Instance_Best_Distance-Best_Known_Result, ((double)clock()-Current_Instance_Begin_Time)/CLOCKS_PER_SEC); 
	int Cur_City=Start_City;
	do
	{
		fprintf(fp,"%d ",Cur_City+1);
		Cur_City=All_Node[Cur_City].Next_City;		
	}while(Cur_City != Null && Cur_City != Start_City);
	
	fprintf(fp,"\n"); 	
	fclose(fp); 
	
	printf("\n%s \t City Num: %d \t Best_known: %d \t MCTS: %d \t Gap: %d \t Time: %.2f Seconds\n",	Input_Inst_Name, Virtual_City_Num, Best_Known_Result,Current_Instance_Best_Distance, 
			Current_Instance_Best_Distance-Best_Known_Result, ((double)clock()-Current_Instance_Begin_Time)/CLOCKS_PER_SEC); 
				
	Release_Memory(Virtual_City_Num);	
}

bool Solve_Instances_In_Batch()
{ 
	ifstream FIC;
	FIC.open(Input_Inst_File_Name);  
  
	if(FIC.fail())
	{
    	cout << "\n\nError! Fail to open file"<<Input_Inst_File_Name<<endl;
    	getchar();
    	return false;     
	}
  	else
    	cout << "\n\nRead instances information from "<<Input_Inst_File_Name<<endl;
      	    
  	FIC>>Test_Inst_Num;     
  	cout<<"Number of Instances: " <<Test_Inst_Num <<endl;  
		  	
	FILE *fp;   
	fp=fopen(Statistics_File_Name, "w+");     
	fprintf(fp,"%d\n",Test_Inst_Num);  
	fclose(fp);   
 
 	Distance_Type Temp_Best_Known;
  	for(int i=0;i<Test_Inst_Num;i++)   
  	{
  		FIC>>&Instance_Name[i][0];  
  		FIC>>Temp_Best_Known;  
  		Best_Known[i]=Temp_Best_Known;
	}      
  	FIC.close();    

  	for(int i=0;i<Test_Inst_Num;i++)
  	{
  	   	strcpy(Input_Inst_Name,&Instance_Name[i][0]); 
    	Best_Known_Result = Best_Known[i];    
		Solve_One_Instance();	
  	}
	     
  	return true;  
}

int main(int argc, char ** argv)
{  	
	double Overall_Begin_Time=(double)clock();
	
	srand(Random_Seed); 
			
	Solve_Instances_In_Batch();	
	
	FILE *fp; 
	fp=fopen(Statistics_File_Name, "a+");  	
	fprintf(fp,"\nMatch_Best_Known_Times: %d Miss_Best_Known_Times: %d Avg gap: %f  Total time:%.2f Seconds\n", Match_Best_Known_Times, Miss_Best_Known_Times, Sum_Gap/Test_Inst_Num,  ((double)clock()-Overall_Begin_Time)/CLOCKS_PER_SEC);
	fclose(fp);
	
	printf("\nMatch_Best_Known_Times: %d Miss_Best_Known_Times: %d Avg gap: %f Total_time:%.2f Seconds\n", Match_Best_Known_Times, Miss_Best_Known_Times, Sum_Gap/Test_Inst_Num,  ((double)clock()-Overall_Begin_Time)/CLOCKS_PER_SEC);
	getchar();
		
	return 0;
}
*/ 




