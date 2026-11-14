// -----------------------------------------------------------------------------
// pe.v - Processing element for an output-stationary systolic array
//
// Every cycle the PE multiplies the two operands it sees, adds the product to
// its local accumulator, and forwards both operands (registered) to its right
// and bottom neighbours. The accumulator never moves: it holds C[i][j].
//
// Operands outside the valid window are driven as 0 by the controller, so no
// enable signal is needed - zeros simply add nothing.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module pe #(
    parameter DW = 8,    // operand width (int8)
    parameter AW = 32    // accumulator width (int32)
) (
    input  wire                 clk,
    input  wire                 rst,    // synchronous reset
    input  wire                 clear,  // synchronous clear before each run
    input  wire signed [DW-1:0] a_in,   // from left neighbour / A edge
    input  wire signed [DW-1:0] b_in,   // from top neighbour / B edge
    output reg  signed [DW-1:0] a_out,  // to right neighbour
    output reg  signed [DW-1:0] b_out,  // to bottom neighbour
    output reg  signed [AW-1:0] acc     // C[i][j]
);
    always @(posedge clk) begin
        if (rst || clear) begin
            a_out <= {DW{1'b0}};
            b_out <= {DW{1'b0}};
            acc   <= {AW{1'b0}};
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            acc   <= acc + a_in * b_in;   // signed multiply-accumulate
        end
    end
endmodule
