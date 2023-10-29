terminal -> hbase shell
create 'register','accno','name','password','email','age'
list
disable 'register'
is_disabled 'register'
enable 'register'
describe 'register'
alter 'f1', NAME=> 'name', VERSION => 5
exists 'register'
drop 'register'
#insert rows
create 'register','personal data','account data'
put 'register','1','personal data:name','raj'
put 'register','1','personal data:age','11'
put 'register','1','personal data:email','raj@gmail.com'
put 'register','1','personal data:accno','1'
scan 'register'
#update
put 'register','1','personal data:age','18'
get 'register','2'
delete 'registter','1','personal data:name',1661584013135
count 'register'
truncate 'register'
