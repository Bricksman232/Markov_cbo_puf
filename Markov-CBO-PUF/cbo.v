module cbo#(
  parameter nStage=64,
  parameter K=2,
  parameter M=2
)(
  input [nStage-1:0] challenge_in,
  input [K-1:0] feedback_in,
  output [nStage-1:0] challenge_out
);
  
  localparam chain_len = nStage/M;

  wire [nStage-1:0] inner_vector;
  wire [M-1:0] init_obf_vector;
  

  //stage 0:initial obfuscation
  assign init_obf_vector = {challenge_in[nStage-1 -: K],feedback_in[K-1:0]};
  assign inner_vector[M-1:0] = init_obf_vector ^ challenge_in[M-1:0];

  genvar i;
  generate
    for (i=1; i<chain_len; i=i+1) begin: Obfucasted_group
      assign inner_vector[ M*i +: M] = challenge_in[ M*i +: M] ^ inner_vector[ M*(i-1) +: M];
    end
  endgenerate

  assign challenge_out = inner_vector;
  
endmodule
