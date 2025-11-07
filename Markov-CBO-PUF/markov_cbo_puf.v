module markov_cbo_puf #(parameter N=16,K=8)(clk,tigSignal,c,respReady,respBitA,respBit);

	input clk;
	input tigSignal;
	input [N-1:0] c;
	output respReady;
	output [K-1:0] respBitA;
	output respBit;
	//output [K-1:0] respBit_f;
	//output resp_xor;
	
	wire [K-1:0] respReadyA;
	wire [K-1:0] respBit_f;
	wire [N-1:0] c_real;
	//wire resp_xor;
	//wire up,down;
	//wire [15:0] test;
	//wire [K-2:0] sel_resp;
	wire resp_choose;

	assign respReady = &respReadyA;    // When response is ready for sampling
	assign resp_choose = ^respBit_f;
	//assign respBit=^respBitA;
	//assign test = {respBit_f,c[63:50]};
	//assign resp_xor = ^respBitA;

	
//  	assign sel_resp[0] = respBit_f[0]^respBit_f[1];
//  	assign sel_resp[1] = respBit_f[2]^respBit_f[3];
//  	assign sel_resp[2] = up^down;
// //
//  	assign up = (sel_resp[0])? respBitA[1] : respBitA[0];
//  	assign down = (sel_resp[1])? respBitA[3] : respBitA[2];
//  	assign respBit = (sel_resp[2])? down : up;
 	//assign selresp = sel_resp[0];

	assign respBit = (resp_choose)? respBitA[1] : respBitA[0];

	/*resp resp_u1(
	 .respBitA(respBitA[3:0]),
	 .resp_choose(respBit_f[3:0]),
	 .respBit(respBit)
	);*/
	(* KEEP_HIERARCHY = "TRUE" *)
	cbo chl_cbo(
	 .challenge_in(c),
	 .feedback_in(respBit_f),
	 .challenge_out(c_real)
	);
	
	
	// APUF 
	
	genvar i;
	generate
		for(i=0; i<K; i=i+1) begin: PUFList
		
			(* KEEP_HIERARCHY = "TRUE" *)
			apufClassic #(.nStage(N)) APUF(
				.clk(clk), 
				.tigSignal(tigSignal), 
				.vcc(1'b1),
				.c(c_real),
				.respReady(respReadyA[i]), 
				.respBit(respBitA[i]),
				.respBit_f(respBit_f[i])
			);
			
		end
	endgenerate
	
		
endmodule
