`timescale 1ns/1ps

module tb_top_1;
localparam WIDTH = 32;

reg        clk; // clk
reg        rst_n; // reset

initial begin
    clk = 1'b0;
    forever #0.5 clk = ~clk;
end
initial begin
    rst_n = 1'b0;
    #10 rst_n = 1'b1;
end
initial begin
//    $fsdbDumpfile("test.fsdb");
//    $fsdbDumpvars(0, tb_top);
end
initial begin
    #10000;
    $finish;
end

reg [7  :0] inst[3:0];
reg [511:0] neuron[139:0];
reg [511:0] weight[139:0];
reg [ 44:0] result[3:0];

initial
begin
  $readmemh("D:/pe_exp/data/inst", inst);
  $readmemh("D:/pe_exp/data/neuron", neuron);
  $readmemh("D:/pe_exp/data/weight", weight);
  $readmemb("D:/pe_exp/data/result", result);
end

reg [ 1:0]   inst_addr;
reg [ 7:0]   iter;
reg [15:0]   neuron_addr;
reg [15:0]   weight_addr;

wire [  7:0] pe_inst   = inst[inst_addr];
wire [511:0] pe_weight = weight[weight_addr];
wire [511:0] pe_neuron = neuron[neuron_addr];

wire [1:0] pe_ctl;
assign pe_ctl[0] = (iter[7:0] == 8'h0);
assign pe_ctl[1] = (iter[7:0] == (pe_inst[7:0] - 1'b1));

reg pe_vld_i;
always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    pe_vld_i <= 1'b0;
  end else if((inst_addr == 2'h0) && (neuron_addr == 16'h0) && (weight_addr == 16'h0)) begin
    pe_vld_i <= 1'b1;
  end else if((inst_addr == 2'h1) && (neuron_addr == 16'h14) && (weight_addr == 16'h14)) begin
    pe_vld_i <= 1'b1;
  end else if((inst_addr == 2'h2) && (neuron_addr == 16'h32) && (weight_addr == 16'h32)) begin
    pe_vld_i <= 1'b1;
  end else if((inst_addr == 2'h3) && (neuron_addr == 16'h5a) && (weight_addr == 16'h5a)) begin
    pe_vld_i <= 1'b1;
  end else if(pe_ctl[1] && pe_vld_i && (inst_addr != 2'h3)) begin
    pe_vld_i <= 1'b0;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    iter <= 8'b0;
  end else if(pe_ctl[1] && pe_vld_i && (inst_addr != 2'h3)) begin
    iter <= 8'b0;
  end else if(pe_vld_i) begin
    iter <= iter + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    inst_addr <= 2'b0;
  end else if(pe_ctl[1] && pe_vld_i && (inst_addr != 2'h3)) begin
    inst_addr <= inst_addr + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    weight_addr <= 16'h0;
  end else if(pe_vld_i) begin
    weight_addr <= weight_addr + 1'b1;

  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    neuron_addr <= 16'h0;
  end else if(pe_vld_i) begin
    neuron_addr <= neuron_addr + 1'b1;
  end
end

wire [31:0] pe_result;
wire pe_vld_o;
parallel_pe u_parallel_pe (
  .clk                  (clk      ),
  .rst_n                (rst_n    ),
  .neuron               (pe_neuron),
  .weight               (pe_weight),
  .ctl                  (pe_ctl   ),
  .vld_i                (pe_vld_i ),
  .result               (pe_result),
  .vld_o                (pe_vld_o )
);

reg [1:0] result_addr;
reg compare_pass;
always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    result_addr <= 2'h0;
  end else if(pe_vld_o) begin
    result_addr <= result_addr + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    compare_pass <= 1'b1;
  end else if(pe_vld_o && (pe_result != result[result_addr][31:0])) begin
    $display("FAIL: num.%d result not correct!!!", result_addr);
    compare_pass <= 1'b0;
  end else if(pe_vld_o && (pe_result == result[result_addr][31:0])) begin
    $display("INFO: num.%d result is correct.", result_addr);
  end
end

endmodule
