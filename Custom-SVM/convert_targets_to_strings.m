function [y_train_str,y_test_str] = convert_targets_to_strings(y_train,y_test)

y_train_str = strrep(cellstr(num2str(y_train)),"-1","Unknown");
y_test_str =  strrep(cellstr(num2str(y_test)),"-1","Unknown");
y_train_str = strrep(y_train_str," 0","Non-Phishing Event");
y_test_str =   strrep(y_test_str," 0","Non-Phishing Event");
y_train_str = strrep(y_train_str," 1","Phishing Event");
y_test_str =   strrep(y_test_str," 1","Phishing Event");

end