function [y_str,y_train_str,y_test_str] = convert_targets_to_strings(y,y_train,y_test)

y_train_str = strrep(cellstr(num2str(y_train)),"-1","Unknown");
y_test_str =  strrep(cellstr(num2str(y_test)),"-1","Unknown");
y_str =  strrep(cellstr(num2str(y)),"-1","Unknown");
y_train_str = strrep(y_train_str," 0","Non-Phishing Event");
y_test_str =   strrep(y_test_str," 0","Non-Phishing Event");
y_str =   strrep(y_str," 0","Non-Phishing Event");
y_train_str = strrep(y_train_str," 1","Phishing Event");
y_test_str =   strrep(y_test_str," 1","Phishing Event");
y_str =   strrep(y_str," 1","Phishing Event");

end