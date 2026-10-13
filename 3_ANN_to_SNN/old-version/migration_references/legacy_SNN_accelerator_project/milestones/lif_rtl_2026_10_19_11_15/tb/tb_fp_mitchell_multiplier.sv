`timescale 1ns/1ps

module tb_fp_mitchell_multiplier;
  logic [31:0] a32;
  logic [31:0] b32;
  logic [31:0] result32;
  logic [15:0] a16;
  logic [15:0] b16;
  logic [15:0] result16;
  logic [31:0] expected32;
  logic [15:0] expected16;
  integer file32;
  integer file16;
  integer scan_result;
  integer vector_count32;
  integer vector_count16;

  fp_mitchell_multiplier #(.EXP_WIDTH(8), .FRAC_WIDTH(23)) dut32 (
    .a_i(a32), .b_i(b32), .result_o(result32)
  );

  fp_mitchell_multiplier #(.EXP_WIDTH(5), .FRAC_WIDTH(10)) dut16 (
    .a_i(a16), .b_i(b16), .result_o(result16)
  );

  task automatic check32(input logic [31:0] a, b, expected);
    begin
      a32 = a;
      b32 = b;
      #1;
      if (result32 !== expected)
        $fatal(1, "FP32 mismatch a=%h b=%h got=%h expected=%h", a, b, result32, expected);
    end
  endtask

  task automatic check16(input logic [15:0] a, b, expected);
    begin
      a16 = a;
      b16 = b;
      #1;
      if (result16 !== expected)
        $fatal(1, "FP16 mismatch a=%h b=%h got=%h expected=%h", a, b, result16, expected);
    end
  endtask

  initial begin
    a32 = '0;
    b32 = '0;
    a16 = '0;
    b16 = '0;

    // 1.5 * 1.5: exact 2.25, Mitchell approximation 2.0.
    check32(32'h3fc00000, 32'h3fc00000, 32'h40000000);
    check16(16'h3e00, 16'h3e00, 16'h4000);

    // Powers of two remain exact.
    check32(32'h40000000, 32'h3f000000, 32'h3f800000);
    check16(16'h4000, 16'h3800, 16'h3c00);

    // Sign and special-case handling.
    check32(32'hbfc00000, 32'h3fc00000, 32'hc0000000);
    check16(16'hbe00, 16'h3e00, 16'hc000);
    check32(32'h00000000, 32'h7f800000, 32'h7fc00000);
    check16(16'h0000, 16'h7c00, 16'h7e00);
    check32(32'h00000001, 32'h3f800000, 32'h00000000);
    check32(32'h7f7fffff, 32'h40000000, 32'h7f800000);

    vector_count32 = 0;
    file32 = $fopen("vectors/fp32_mitchell_vectors.mem", "r");
    if (file32 == 0)
      $fatal(1, "cannot open FP32 golden vectors");
    while (!$feof(file32)) begin
      scan_result = $fscanf(file32, "%h %h %h\n", a32, b32, expected32);
      if (scan_result == 3) begin
        #1;
        if (result32 !== expected32)
          $fatal(1, "FP32 vector %0d got=%h expected=%h", vector_count32, result32, expected32);
        vector_count32 = vector_count32 + 1;
      end
    end
    $fclose(file32);

    vector_count16 = 0;
    file16 = $fopen("vectors/fp16_mitchell_vectors.mem", "r");
    if (file16 == 0)
      $fatal(1, "cannot open FP16 golden vectors");
    while (!$feof(file16)) begin
      scan_result = $fscanf(file16, "%h %h %h\n", a16, b16, expected16);
      if (scan_result == 3) begin
        #1;
        if (result16 !== expected16)
          $fatal(1, "FP16 vector %0d got=%h expected=%h", vector_count16, result16, expected16);
        vector_count16 = vector_count16 + 1;
      end
    end
    $fclose(file16);

    $display("PASS tb_fp_mitchell_multiplier fp32_vectors=%0d fp16_vectors=%0d",
      vector_count32, vector_count16);
    $finish;
  end
endmodule
