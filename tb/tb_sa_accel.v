// -----------------------------------------------------------------------------
// tb_sa_accel.v - Self-checking testbench for sa_accel
//
// Driven by scripts/run_rtl_tests.py, which writes build/{a,b,c}.hex and passes
// the inner dimension as +K=<n>. The testbench loads A and B over the register
// bus, starts the run, checks the cycle counter equals K + 2N - 2, and compares
// every C element against the Python reference. Prints PASS/FAIL.
//
//   hex files: a.hex = A row-major (N x K), b.hex = B row-major (K x N),
//              c.hex = expected C row-major (N x N, 32-bit)
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_sa_accel;
    parameter N    = 4;
    parameter KMAX = 64;

    reg         clk = 1'b0;
    reg         rst = 1'b1;
    reg         bus_we = 1'b0;
    reg  [15:0] bus_addr = 16'd0;
    reg  [31:0] bus_wdata = 32'd0;
    wire [31:0] bus_rdata;

    sa_accel #(.N(N), .KMAX(KMAX)) dut (
        .clk(clk), .rst(rst),
        .bus_we(bus_we), .bus_addr(bus_addr),
        .bus_wdata(bus_wdata), .bus_rdata(bus_rdata)
    );

    always #5 clk = ~clk;

    reg [7:0]  a_h [0:N*KMAX-1];
    reg [7:0]  b_h [0:KMAX*N-1];
    reg [31:0] c_h [0:N*N-1];

    integer K, i, j, k, idx, errors, polls;
    reg [31:0] rd;

    task bus_write(input [15:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            bus_addr  = addr;
            bus_wdata = data;
            bus_we    = 1'b1;
            @(negedge clk);
            bus_we    = 1'b0;
        end
    endtask

    task bus_read(input [15:0] addr);
        begin
            bus_addr = addr;
            #1 rd = bus_rdata;
        end
    endtask

    initial begin
        if (!$value$plusargs("K=%d", K)) K = N;
        if ($test$plusargs("vcd")) begin
            $dumpfile("build/wave.vcd");
            $dumpvars(0, tb_sa_accel);
        end

        $readmemh("build/a.hex", a_h);
        $readmemh("build/b.hex", b_h);
        $readmemh("build/c.hex", c_h);

        errors = 0;
        repeat (3) @(negedge clk);
        rst = 1'b0;

        // sanity: build parameters visible over the bus
        bus_read(16'h0004); if (rd !== N)    begin errors = errors + 1; $display("FAIL: N reg = %0d", rd); end
        bus_read(16'h0005); if (rd !== KMAX) begin errors = errors + 1; $display("FAIL: KMAX reg = %0d", rd); end

        // load operands
        bus_write(16'h0002, K);
        for (i = 0; i < N; i = i + 1)
            for (k = 0; k < K; k = k + 1)
                bus_write(16'h1000 + i*KMAX + k, {24'd0, a_h[i*K + k]});
        for (k = 0; k < K; k = k + 1)
            for (j = 0; j < N; j = j + 1)
                bus_write(16'h2000 + k*N + j, {24'd0, b_h[k*N + j]});

        // start and wait for done (status bit1)
        bus_write(16'h0000, 32'd1);
        polls = 0;
        rd = 32'd0;
        while (!rd[1] && polls < 10*(K + 2*N) + 100) begin
            @(negedge clk);
            bus_read(16'h0001);
            polls = polls + 1;
        end
        if (!rd[1]) begin errors = errors + 1; $display("FAIL: timeout waiting for done"); end
        if (rd[2])  begin errors = errors + 1; $display("FAIL: error flag set"); end

        // cycle counter must equal K + 2N - 2
        bus_read(16'h0003);
        if (rd !== K + 2*N - 2) begin
            errors = errors + 1;
            $display("FAIL: cycles = %0d, expected %0d", rd, K + 2*N - 2);
        end

        // compare every C element
        for (idx = 0; idx < N*N; idx = idx + 1) begin
            bus_read(16'h3000 + idx);
            if (rd !== c_h[idx]) begin
                errors = errors + 1;
                $display("FAIL: C[%0d][%0d] = %0d, expected %0d",
                         idx / N, idx % N, $signed(rd), $signed(c_h[idx]));
            end
        end

        if (errors == 0) $display("PASS N=%0d K=%0d cycles=%0d", N, K, K + 2*N - 2);
        else             $display("FAIL N=%0d K=%0d errors=%0d", N, K, errors);
        $finish;
    end
endmodule
